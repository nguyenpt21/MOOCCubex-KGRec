
import setproctitle
import random
from tqdm import tqdm
import torch
import numpy as np
import os
from time import time
from prettytable import PrettyTable
import datetime
from utils.parser import parse_args_kgsr
from utils.data_loader import load_data
from modules.KGRec import KGRec
from utils.evaluate_kgsr import test
from utils.helper import *
from logging import getLogger
from utils.sampler import UniformSampler
from collections import defaultdict

seed = 2020
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "utils/ext/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(seed)
except:
    sampling = UniformSampler(seed)

setproctitle.setproctitle('EXP@KGRec')

def neg_sampling_cpp(train_cf_pairs, train_user_dict):
    time1 = time()
    train_cf_negs = sampling.sample_negative(train_cf_pairs[:, 0], n_items, train_user_dict, 1)
    train_cf_negs = np.asarray(train_cf_negs)
    train_cf_triples = np.concatenate([train_cf_pairs, train_cf_negs], axis=1)
    time2 = time()
    logging.info('neg_sampling_cpp time: %.2fs', time2 - time1)
    logging.info('train_cf_triples shape: {}'.format(train_cf_triples.shape))
    return train_cf_triples

def get_feed_dict(train_cf_with_neg, start, end):
    feed_dict = {}
    entity_pairs = torch.from_numpy(train_cf_with_neg[start:end]).to(device).long()
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = entity_pairs[:, 2]
    feed_dict['batch_start'] = start
    return feed_dict

if __name__ == '__main__':
    try:
        """fix the random seed"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        """read args"""
        global args, device
        args = parse_args_kgsr()
        device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

        log_save_id = create_log_id(args.out_dir)
        logging_config(folder=args.out_dir, name='log{:d}'.format(log_save_id), no_console=False)

        logging.info(args)
        

        """build dataset"""
        train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
        adj_mat_list, norm_mat_list, mean_mat_list = mat_list

        n_users = n_params['n_users']
        n_items = n_params['n_items']
        n_entities = n_params['n_entities']
        n_relations = n_params['n_relations']
        n_nodes = n_params['n_nodes']

        """define model"""
        model_dict = {
            'KGSR': KGRec,
        }
        model = model_dict[args.model]
        model = model(n_params, args, graph, mean_mat_list[0]).to(device)
        model.print_shapes()
        """define optimizer"""
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        test_interval = args.test_interval
        early_stop_step = args.stop_steps 

        cur_best_pre = 0
        cur_stopping_step = 0
        should_stop = False

        logging.info("start training ...")
        for epoch in range(1, args.epoch + 1):
            """training CF"""
            """cf data"""
            train_cf_with_neg = neg_sampling_cpp(train_cf, user_dict['train_user_set'])
            # shuffle training data
            index = np.arange(len(train_cf))
            np.random.shuffle(index)
            train_cf_with_neg = train_cf_with_neg[index]

            """training"""
            model.train()
            add_loss_dict, s = defaultdict(float), 0
            train_s_t = time()
            with tqdm(total=len(train_cf)//args.batch_size) as pbar:
                while s + args.batch_size <= len(train_cf):
                    batch = get_feed_dict(train_cf_with_neg,
                                        s, s + args.batch_size)
                    batch_loss, batch_loss_dict = model(batch)

                    optimizer.zero_grad(set_to_none=True)
                    batch_loss.backward()
                    optimizer.step()

                    for k, v in batch_loss_dict.items():
                        add_loss_dict[k] += v
                    s += args.batch_size
                    pbar.update(1)

            train_e_t = time()

            if epoch % test_interval == 0:
                """testing"""
                print('testing')
                test_s_t = time()
                model.eval()
                with torch.no_grad():
                    ret = test(model, user_dict, n_params)
                test_e_t = time()

                train_res = PrettyTable()
                train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, list(add_loss_dict.values()), ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
                )
                logging.info(train_res)

                # *********************************************************
                # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
                cur_best_pre, cur_stopping_step, should_stop = early_stopping(ret['recall'][-1], cur_best_pre,cur_stopping_step, expected_order='acc', flag_step=early_stop_step)
                if cur_stopping_step == 0:
                    logging.info("###find better!")
                elif should_stop:
                    break
                """save weight"""
                if ret['recall'][-1] == cur_best_pre and args.save:
                    save_path = args.out_dir + 'model_epoch{}.pth'.format(epoch)
                    logging.info('save better model at epoch %d to path %s' % (epoch, save_path))
                    checkpoint_path = args.out_dir + 'checkpoint.pth'
                    torch.save(model.state_dict(), save_path)
                    torch.save({
                        'epoch': epoch,
                        'best_recall': cur_best_pre,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'cur_stopping_step': cur_stopping_step
                    }, checkpoint_path)

            else:
                # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
                logging.info('{}: using time {}, training loss at epoch {}: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), train_e_t - train_s_t, epoch, list(add_loss_dict.values())))

        logging.info('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre))

    except Exception as e:
        logging.exception(e)