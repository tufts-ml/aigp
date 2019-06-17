import tensorflow as tf
import numpy as np


class VanillaEnt:
    def __init__(self, model):
        # initialize model
        self.model = model
        # initialize variational parameters
        n_locations = self.model.n_Xtrain
        n_parents = self.model.hyper_param['n_parents']
        self.n_parents = n_parents
        self.variational = {'mu': tf.Variable(0.5 * tf.ones([n_locations, 1], dtype=tf.float64)),
                            'L': tf.Variable(tf.random_uniform([n_locations, n_parents + 1], minval=1.0, maxval=2.0,
                                                               dtype=tf.float64))
                            }

        self.std_normal = tf.contrib.distributions.Normal(loc=tf.to_double(0.0),
                                                          scale=tf.to_double(1.0))

    def gather_L(self, indices):
        L = tf.gather(self.variational['L'], indices)
        return L

    def gather_mu(self, indices):
        mu = tf.gather(self.variational['mu'], indices)
        return mu

    def sample(self, point_index):
        T = self.model.cons['T']
        m_k = self.gather_mu(point_index)
        parent_k = tf.gather(self.model.cons['parents'], point_index)
        parent_k = tf.concat([tf.ones([1, 1], dtype=tf.int32), parent_k], axis=1)
        L_k = self.gather_L(point_index)
        L_k = L_k * tf.cast(tf.clip_by_value(parent_k, 0, 1), dtype=tf.float64)
        epsilon = self.std_normal.sample([1 + self.n_parents, T])
        samples = tf.matmul(L_k, epsilon) + m_k
        return samples

    def cal_entropy_term(self, indices):
        # only work for L[:, 0] is L_ii!!!!
        L = self.gather_L(indices)
        L_ii = tf.gather(L, [0], axis=1)
        log_det = 0.5 * tf.reduce_mean(tf.log(tf.square(L_ii)))
        cons = 0.5 * np.log(2.0 * np.pi * np.exp(1.0))
        entropy = cons + log_det
        return entropy
