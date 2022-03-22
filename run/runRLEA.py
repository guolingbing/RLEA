# IMPORTS
import os
import argparse
import sys
import time
import logging
from copy import copy, deepcopy


import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from openea.modules.args.args_hander import check_args, load_args
from openea.modules.load.kgs import read_kgs_from_folder
from openea.modules.bootstrapping.alignment_finder import find_alignment

from openea.approaches import JAPE
from openea.approaches import SEA
from openea.approaches import RSN4EA
from openea.approaches import RDGCN
from openea.approaches import BootEA
from openea.approaches import AlignE

from rlea.agent import Policy, Baseline
from rlea.environment import EntityAlignmentEnvironment
from rlea.reinforcement import reinforce

parser = argparse.ArgumentParser(description='Reinforment Learning for Entity Alignment')
parser.add_argument('--dataset', type=str, default='D_Y')
parser.add_argument('--model_name', type=str, default='rdgcn')
parser.add_argument('--mapping', type=bool, default=False,
 help='if mapping, project the embeddings of target KG to source KG')
parser.add_argument('--restore_embeddings', type=bool, default=False,
 help='restore the trained embeddings')

parser.add_argument('--action_size', type=int, default=2)
parser.add_argument('--candidate_num', type=int, default=10)
parser.add_argument('--num_episodes', type=int, default=500)
parser.add_argument('--discount_rate', type=int, default=.95)

parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--policy_lr', type=float, default=1e-4)
parser.add_argument('--MIE', type=bool, default=True)

parser.add_argument('--random', type=bool, default=False)
parser.add_argument('--skip_rate', type=float, default=.5)
parser.add_argument('--skip_discount_rate', type=float, default=.99)
parser.add_argument('--min_skip_rate', type=float, default=.2)



class ModelFamily(object):
    jape = JAPE
    aligne = AlignE
    bootea = BootEA
    sea = SEA
    rdgcn = RDGCN

def get_model(model_name):
    return getattr(ModelFamily, model_name)

def init_logger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


    LOG_FORMAT = '%(asctime)s -- %(levelname)s#: %(message)s '
    DATE_FORMAT = '%Y-%m-%d  %H:%M:%S %a'
    LOG_PATH = 'logs/%s.log' % path


    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    
    return logger

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN gnn and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)



if __name__ == '__main__':
    rlea_args = parser.parse_args()
    model_name = rlea_args.model_name
    dataset = rlea_args.dataset
    save_name = '%s_%s' % (model_name,dataset)

    logger = init_logger(save_name)
    logger.info(rlea_args)


    logger.info('\nInit OpenEA framework\n')
    
    args = load_args('./args/%s_args_15K.json' % model_name)
    args.training_data = '%s%s_15K_V1/' % (args.training_data, dataset)
    args.dataset_division = '721_5fold/1/'

    remove_unlinked = False
    if args.embedding_module == "RSN4EA":
        remove_unlinked = True
    kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                            remove_unlinked=remove_unlinked)

    model = get_model(model_name)()
    model.set_args(args)
    model.set_kgs(kgs)
    model.init()

    if model.ent_embeds is None:
        model.ent_embeds = model.output
    if model.session is None:
        model.session = model.sess

    # Training or Restore the EEA model
    saver = tf.train.Saver()
    ckpt_path = 'ckpts/%s_V1_1.ckpt' % save_name.upper()
    if rlea_args.restore_embeddings:
        logger.info('Restore the EEA model from ckpt')
        saver.restore(model.session, ckpt_path)
    else:
        logger.info('Start to train the EEA model')
        model.run()
        saver.save(model.session, ckpt_path)


    # RLEA
    logger.info('RLEA: init adj matrix')
    all_triples = np.concatenate(
    [kgs.kg1.relation_triples_list, kgs.kg2.relation_triples_list])
    all_edges = np.concatenate([all_triples[:, [0, 2]], all_triples[:, [2, 0]]])
    all_edges_df = pd.DataFrame(all_edges, columns=['h_id', 't_id'])
    uniqued = all_edges_df.groupby(['h_id','t_id']).size().reset_index()
    adj = sp.coo_matrix((np.ones_like(uniqued.values[:, -1]), (uniqued.values[:, 0],
                                                            uniqued.values[:, 1])), shape=(kgs.entities_num, kgs.entities_num))
    adj = preprocess_adj(adj)


    rlea_args.adj = adj
    rlea_args.kgs = kgs
    rlea_args.dim = args.dim
    rlea_args.logger = logger


    logger.info('RLEA: init environments')
    env = EntityAlignmentEnvironment(model, rlea_args, kgs.train_links)
    valid_env = EntityAlignmentEnvironment(model, rlea_args, kgs.valid_links, is_test=True)
    test_env = EntityAlignmentEnvironment(model, rlea_args, kgs.test_links, is_test=True)

    logger.info('RLEA: init policy networks')
    g1 = tf.Graph()
    tf.reset_default_graph()
    with g1.as_default():
        global_step = tf.Variable(0, name="global_step", trainable=False)
        policy_estimator = Policy(rlea_args)
        value_estimator = Baseline(rlea_args)
        
    # init sess
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=g1, config=config)
    rlea_args.sess = sess

    logger.info('RLEA: start to train')
    with g1.as_default():
        sess.run(tf.global_variables_initializer())
        reinforce(sess, env, valid_env, test_env,  rlea_args, policy_estimator, value_estimator)
