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

sigmod = lambda x: 1.0/(1+pow(math.e, -x))
sigmod_der = lambda x: pow(math.e, x) / (1 + pow(math.e, x)) ** 2
sigmod_f = np.vectorize(lambda x: sigmod(x))
sigmod_d = np.vectorize(lambda x: sigmod_der(x))


class PMF(object):

    def __init__(self):
        init_logger(log_file='log/pmf.log', log_level=logging.INFO)
        self.ratings_file = ratings_file
        self.load_data()
        self.obs_num = self.ratings_vector.shape[0]

        self.generate_normalized_ratings()
        rand_inds = np.random.permutation(self.obs_num)
        train_ratio = 0.8
        self.train_num = int(self.obs_num * train_ratio)

        self.train_vector = self.ratings_vector[rand_inds[:self.train_num]]
        self.vali_vector = self.ratings_vector[rand_inds[self.train_num:]]
        logging.info('observations=%s, train_ratio=%s, train_num=%s, vali_num=%s',\
                self.obs_num, train_ratio, self.train_vector.shape[0], self.vali_vector.shape[0])
        del rand_inds

        self.epsilon = 0.5; #learning rate
        self.lamb = 0.01 #Regularization parameter
        self.momentum = 0.8
        self.max_epoch = 50
        self.feat_num = 5

        #uid, vid以observation里出现的uid为准, 如何划分数据也是一个问题
        self.user_num = self.ratings_vector[:,0].max()
        self.item_num = self.ratings_vector[:,1].max()

        self.U_shape = (self.user_num, self.feat_num)
        self.V_shape = (self.item_num, self.feat_num)
        #U: matrix of user features, V: matrix of item features
        self.U = 0.1 * np.random.standard_normal(self.U_shape)
        self.V = 0.1 * np.random.standard_normal(self.V_shape)

        self.U_inc = np.zeros(self.U_shape)
        self.V_inc = np.zeros(self.V_shape)

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
            the standard PMF with gradient descent
        '''
        #starting from 0
        user_inds = self.train_vector[:,0].astype(int) - 1
        item_inds = self.train_vector[:,1].astype(int) - 1
        ratings  = self.train_vector[:,2]


        pre_err, cur_err = 999999999, 999999999

        train_start = time.time()

        for epoch in range(1, self.max_epoch):

            round_start = time.time()

            U_V_pairwise = np.multiply(self.U[user_inds,:], self.V[item_inds,:])

            #####compute predictions####
            pred_out = sigmod_f(U_V_pairwise.sum(axis=1))#|R| * K --> |R| * 1

            err_f = 0.5 * (np.square(pred_out - ratings).sum() + 0.5 * self.lamb * (np.square(self.U).sum() + np.square(self.V).sum()))

            pre_err = cur_err
            cur_err = err_f

            #if pre_err - cur_err < 10:
            #    break

            pred_time = time.time()
            #####calculate the gradients#####

            grad_u = np.zeros(self.U_shape)
            grad_v = np.zeros(self.V_shape)

            ####update gradient
            #U_mat = np.zeros((self.user_num, self.tr_num))
            #V_mat = np.zeros((self.item_num, self.tr_num))
            sigmod_dot = sigmod_f(U_V_pairwise.sum(axis=1))
            sigmod_der_V = sigmod_d(U_V_pairwise.sum(axis=1))
            U_V_delta = np.multiply(sigmod_der_V, (sigmod_dot - ratings)).reshape(self.train_num, 1)
            delta_matrix = np.tile(U_V_delta, self.feat_num)
            delta_U = np.multiply(delta_matrix, self.V[item_inds,:])
            delta_V = np.multiply(delta_matrix, self.U[user_inds,:])
            dot_time = time.time()


            '''
            sparse_V_mat = sparse.csr_matrix(V_mat)
            sparse_delta_V = sparse.csr_matrix(delta_V)
            #bad practice, as U_mat is |U| * |R|, delta_U is |R| * |D|, which makes dot product so expensive
            #when choose just 10000 observations, the time cost is "time detail: 1s/0s/49s/52s"
            #when we use the sparse matrix, the time cost is ""
            time1 = time.time()
            for uid in range(int(self.user_num)):
                U_mat[uid] = np.equal(uid, user_inds).astype(dtype=int)

            time2 = time.time()
            grad_u += np.dot(U_mat, delta_U)

            time3 = time.time()
            for vid in range(int(self.item_num)):
                sparse_V_mat[vid] = np.equal(vid, item_inds).astype(dtype=int)

            time4 = time.time()
            grad_v = np.add(grad_v,sparse_V_mat.dot(sparse_delta_V))

            time5 = time.time()
            print 'time detail: %.1fs/%.1fs/%.1fs/%.1fs' % (time2 - time1, time3-time2, time4-time3, time5-time4)
            '''

            ind = 0
            for uid, vid, r in self.train_vector:
                uid -= 1
                vid -= 1
                grad_u[uid] +=  delta_U[ind]
                grad_v[vid] +=  delta_V[ind]
                ind += 1

            accumulate_time = time.time()

            logging.info('dot/accumulate cost %.1fs/%.1fs', dot_time - pred_time, accumulate_time - dot_time)
            '''
            pred_out = sigmod_f(np.multiply(self.U[user_inds,:], self.V[item_inds,:]).sum(axis=1))#|R| * K --> |R| * 1
            dot_time = 0.0
            calculus_time = 0.0
            add_delta_time = 0.0
            for uid, vid, r in ratings_vector:
                uid -= 1
                vid -= 1

                dot_start = time.time()
                u_v_dot = np.dot(self.U[uid], self.V[vid])
                dot_end = time.time()
                dot_time += (dot_end - dot_start)

                cal_start = time.time()
                first_der = sigmod(u_v_dot)
                second_der = sigmod_der(u_v_dot)
                cal_end = time.time()
                calculus_time += (cal_end - cal_start)

                add_delta_start = time.time()
                delta = second_der * (first_der - r)
                grad_u[uid] +=  delta * self.V[vid]
                grad_v[vid] +=  delta * self.U[uid]
                #grad_v[vid] += (np.dot(self.U[uid], self.V[vid]) - r) * self.U[uid]
                add_delta_end = time.time()
                add_delta_time += (add_delta_end - add_delta_start)

            print 'cost detail: u_v_dot=%.1fs, calculus=%.1fs, add_delta=%.1fs' % (dot_time, calculus_time, add_delta_time)
            '''

            grad_u += self.lamb * self.U
            grad_v += self.lamb * self.V
            cal_grad_time = time.time()

            #####update the U and V vectors
            self.U -= self.epsilon * grad_u / math.sqrt(np.square(grad_u).sum())
            self.V -= self.epsilon * grad_v / math.sqrt(np.square(grad_v).sum())
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

    def run(self):
        self.train()
        self.predict()
        self.evaluate()

    def batch_train(self):

        for epoch in range(1, self.max_epoch):
            rand_inds = np.random.permutation(self.tr_num)
            ratings_vector = self.ratings_vector[rand_inds,:]
            del rand_inds

            for batch in range(1, self.batch_num):
                print 'epoch %s batch %d' % (epoch, batch)
                N = 10000

                aa_p =  ratings_vector[(batch-1)*N:batch*N+1,0]
                aa_m =  ratings_vector[(batch-1)*N:batch*N+1,1]

                rating =  ratings_vector[(batch-1)*N:batch*N+1,2]

                rating = rating - self.mean_rating # set mean to 0 ?

                ####compute predictions####
                pred_out = np.multiply(self.U[aa_u,:], self.V[aa_v,:]).sum(axis=1) # sum by column

                f = np.square(pred_out - rating)

if __name__ == '__main__':
    pmf = PMF()
    pmf.run()

