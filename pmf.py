#coding=utf8 
'''
    实现pmf的代码
    version1: rewrite in python the matlab code given by Ruslan, the path is ../code_BPMF/pmf.m
'''

import math
import time
import numpy as np
from scipy import sparse

import logging
from logging_util import init_logger

ratings_file = '../data/epinions/ver1_ratings_data.txt'
trust_file = '../data/epinions/ver1_trust_data.txt'

remove_sigmod = False

sigmod = lambda x: 1.0/(1+pow(math.e, -x))
sigmod_der = lambda x: pow(math.e, x) / (1 + pow(math.e, x)) ** 2
if remove_sigmod:
    sigmod = lambda x: x
    sigmod_der = lambda x: 1
sigmod_f = np.vectorize(lambda x: sigmod(x))
sigmod_d = np.vectorize(lambda x: sigmod_der(x))

class PMF(object):

    def __init__(self):
        init_logger(log_file='log/pmf.log', log_level=logging.INFO)
        self.ratings_file = ratings_file
        self.load_data()
        self.obs_num = self.ratings_vector.shape[0]

        self.generate_normalized_ratings()
        self.split_data()

        self.epsilon = 0.5; #learning rate
        self.lamb = 0.01 #Regularization parameter
        self.momentum = 0.8
        self.max_epoch = 140 #iteration
        self.feat_num = 5

        #uid, vid以observation里出现的uid为准, 如何划分数据也是一个问题
        self.user_num = self.ratings_vector[:,0].max()
        self.item_num = self.ratings_vector[:,1].max()

        self.U_shape = (self.user_num, self.feat_num)
        self.V_shape = (self.item_num, self.feat_num)

        #U: matrix of user features, V: matrix of item features, generated from gaussian distribution
        self.U = np.random.standard_normal(self.U_shape)
        self.V = np.random.standard_normal(self.V_shape)

    def split_data(self):
        '''
            split the data into two parts: train and validation vector
            choose randomly a proportion of data by train_ratio, the remaining as validation data
        '''
        rand_inds = np.random.permutation(self.obs_num)
        train_ratio = 0.8
        self.train_num = int(self.obs_num * train_ratio)

        self.train_vector = self.ratings_vector[rand_inds[:self.train_num]]
        self.vali_vector = self.ratings_vector[rand_inds[self.train_num:]]
        logging.info('observations=%s, train_ratio=%s, train_num=%s, vali_num=%s',\
                self.obs_num, train_ratio, self.train_vector.shape[0], self.vali_vector.shape[0])
        del rand_inds

    def load_data(self):
        '''
            load triplets(user_id, movie_id, rating)
            make user_id and item_id start from zero
        '''
        self.ratings_vector = np.loadtxt(self.ratings_file,delimiter=' ')

    def generate_normalized_ratings(self):
        '''
            mapping the rating 1,...,K to [0, 1] by the formula r = (x - 1) / (K - 1)
        '''
        max_ = self.ratings_vector[:,2].max()
        self.ratings_vector[:,2] = (self.ratings_vector[:,2] - 1.0) / (max_ - 1)

    def train(self):
        '''
            standard PMF with gradient descent
        '''
        #starting from 0
        user_inds = self.train_vector[:,0].astype(int) - 1
        item_inds = self.train_vector[:,1].astype(int) - 1
        ratings  = self.train_vector[:,2]

        train_start = time.time()
        for epoch in range(1, self.max_epoch):

            round_start = time.time()

            U_V_pairwise = np.multiply(self.U[user_inds,:], self.V[item_inds,:])

            #####compute predictions####
            pred_out = sigmod_f(U_V_pairwise.sum(axis=1))#|R| * K --> |R| * 1

            err_f = 0.5 * (np.square(pred_out - ratings).sum() + self.lamb * (np.square(self.U).sum() + np.square(self.V).sum()))

            pred_time = time.time()
            #####calculate the gradients#####

            grad_u = np.zeros(self.U_shape)
            grad_v = np.zeros(self.V_shape)

            ####update gradient
            sigmod_dot = sigmod_f(U_V_pairwise.sum(axis=1))
            sigmod_der_V = sigmod_d(U_V_pairwise.sum(axis=1))
            U_V_delta = np.multiply(sigmod_der_V, (sigmod_dot - ratings)).reshape(self.train_num, 1) #|R| * 1
            delta_matrix = np.tile(U_V_delta, self.feat_num) # |R| * K, 这样就可以使得u_i和对应的v_j进行dot product, 可以直接使用矩阵运算
            delta_U = np.multiply(delta_matrix, self.V[item_inds,:])
            delta_V = np.multiply(delta_matrix, self.U[user_inds,:])
            dot_time = time.time()

            ind = 0
            for uid, vid, r in self.train_vector:
                uid -= 1
                vid -= 1
                grad_u[uid] +=  delta_U[ind]
                grad_v[vid] +=  delta_V[ind]
                ind += 1

            accumulate_time = time.time()

            logging.info('dot/accumulate cost %.1fs/%.1fs', dot_time - pred_time, accumulate_time - dot_time)

            grad_u += self.lamb * self.U
            grad_v += self.lamb * self.V
            cal_grad_time = time.time()

            #####update the U and V vectors
            self.U -= self.epsilon * grad_u
            self.V -= self.epsilon * grad_v
            round_end = time.time()

            logging.info('iter=%s, train error=%s, cost(gradient/round) %.1fs/%.1fs', \
                    epoch, err_f, cal_grad_time - pred_time, round_end - round_start)

        logging.info('training finished, cost %.2fmin', (time.time() - train_start) / 60.0)

    def predict(self):
        '''
            predict the rating using the dot product of U,V
        '''
        user_inds = self.vali_vector[:,0].astype(int) - 1
        item_inds = self.vali_vector[:,1].astype(int) - 1

        self.predictions = sigmod_f(np.multiply(self.U[user_inds,:], self.V[item_inds,:]).sum(axis=1))

    def evaluate(self):
        '''
            calculate the RMSE&MAE
        '''
        vali_ratings = self.vali_vector[:,2]
        delta = self.predictions - vali_ratings
        mae = np.absolute(delta).sum() / delta.shape[0]
        rmse = math.sqrt(np.square(delta).sum() / delta.shape[0])
        logging.info('evaluations: mae=%.2f, rmse=%.2f', mae, rmse)
        logging.info('config: iters=%s, feat=%s, regularization=%s, learning_rate=%s', self.max_epoch, self.feat_num, self.lamb, self.epsilon)
        if remove_sigmod:
            logging.info('***********remove sigmod functions!!!*************')

    def run(self):
        self.train()
        #logging.info('without training!')
        self.predict()
        self.evaluate()

if __name__ == '__main__':
    pmf = PMF()
    pmf.run()

