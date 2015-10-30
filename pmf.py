#coding=utf8 
'''
    实现pmf的代码
    version1: rewrite in python the matlab code given by Ruslan, the path is ../code_BPMF/pmf.m
'''

import numpy as np
import math
import time

ratings_file = '../data/epinions/ver1_ratings_data.txt'
trust_file = '../data/epinions/ver1_trust_data.txt'

sigmod = lambda x: 1.0/(1+pow(math.e, -x))
sigmod_der = lambda x: pow(math.e, x) / (1 + pow(math.e, x)) ** 2


class PMF(object):

    def __init__(self):
        self.ratings_file = ratings_file
        self.load_data()

        self.epsilon = 0.5; #learning rate
        self.lamb = 0.01 #Regularization parameter
        self.momentum = 0.8
        self.max_epoch = 500

        #self.mean_rating = self.ratings_vector[:,2].mean()
        self.tr_num = self.ratings_vector.shape[0]
        #self.va_num = self.vali_vectors.shape[0]

        self.batch_num = 9
        self.user_num = self.ratings_vector[:,0].max()
        self.item_num = self.ratings_vector[:,1].max()

        self.feat_num = 10
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
        #self.vali_vectors = np.loadtxt(self.vali_file)

    def generate_normalized_ratings(self):
        '''
            mapping the rating 1,...,K to [0, 1] by the formula r = (x - 1) / (K - 1)
        '''
        max_ = self.ratings_vector[:,2].max()
        return (self.ratings_vector[:,2] - 1.0) / (max_ - 1)

    def train(self):
        '''
            the standard PMF with gradient descent
        '''
        #rand_inds = np.random.permutation(self.tr_num)
        #ratings_vector = self.ratings_vector[rand_inds,:]
        ratings_vector = self.ratings_vector
        #del rand_inds

        #starting from 0
        user_inds = ratings_vector[:,0].astype(int) - 1
        item_inds = ratings_vector[:,1].astype(int) - 1

        ratings = self.generate_normalized_ratings()
        ratings_vector[:,2] = ratings


        sigmod_f = np.vectorize(lambda x: 1.0/(1+pow(math.e, -x)))
        #rating = sigmod(rating)
        #print rating[:10]

        pre_err, cur_err = 999999999, 999999999

        train_start = time.time()

        for epoch in range(1, self.max_epoch):

            round_start = time.time()
            #####compute predictions####
            pred_out = sigmod_f(np.multiply(self.U[user_inds,:], self.V[item_inds,:]).sum(axis=1))#|R| * K --> |R| * 1


            err_f = 0.5 * (np.square(pred_out - ratings).sum() + 0.5 * self.lamb * (np.square(self.U).sum() + np.square(self.V).sum()))

            pre_err = cur_err
            cur_err = err_f

            if pre_err - cur_err < 10:
                break

            #####calculate the gradients#####

            grad_u = np.zeros(self.U_shape)
            grad_v = np.zeros(self.V_shape)

            for uid, vid, r in ratings_vector:
                uid -= 1
                vid -= 1

                grad_u[uid] += sigmod_der(np.dot(self.U[uid], self.V[vid])) * (sigmod(np.dot(self.U[uid], self.V[vid])) - r) * self.V[vid]
                grad_v[vid] += sigmod_der(np.dot(self.U[uid], self.V[vid])) * (sigmod(np.dot(self.U[uid], self.V[vid])) - r) * self.U[uid]
                #grad_v[vid] += (np.dot(self.U[uid], self.V[vid]) - r) * self.U[uid]

            grad_u += self.lamb * self.U
            grad_v += self.lamb * self.V

            #####update the U and V vectors
            self.U -= self.epsilon * grad_u / math.sqrt(np.square(grad_u).sum())
            self.V -= self.epsilon * grad_v / math.sqrt(np.square(grad_v).sum())
            round_end = time.time()

            print '%s iterations, train error=%s, cost %.1fs' % (epoch, err_f, round_end - round_start)

        print 'training ended, cost %.2fmin' % ((time.time() - train_start) / 60.0)

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
    pmf.train()

