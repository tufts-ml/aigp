import numpy as np
import tensorflow as tf

import util.util as utils
from inferences.NeuralNet import NeuralNet


class Ent:
    def __init__(self, model):
        # variables for further use
        self.model = model
        self.n_parents = self.model.hyper_param['n_parents']
        self.std_normal = tf.contrib.distributions.Normal(loc=tf.to_double(0.0),
                                                          scale=tf.to_double(1.0))
        # initialize variational parameters
        self.variational = {'mu': NeuralNet(input_dim=1,
                                            hidden_layer_dim=self.model.nn_conf['mu_nn_conf']['hidden_layer_dim'],
                                            activate_func=self.model.nn_conf['mu_nn_conf']['activate_func']),
                            'L': NeuralNet(input_dim=1,
                                           hidden_layer_dim=self.model.nn_conf['L_nn_conf']['hidden_layer_dim'],
                                           activate_func=self.model.nn_conf['L_nn_conf']['activate_func'])}

    def retrieve_ys_DAD(self, indices):
        # retrieve parents and remove the first dim
        parents = tf.gather(self.model.cons['parents'], indices)
        # concatenate index and parents
        indices = tf.concat([tf.expand_dims(indices, axis=-1), parents], axis=1)
        # get ys
        ys = tf.gather(self.model.data['ytrain'], indices)
        ys = tf.expand_dims(ys, axis=-1)
        # get DAD
        xs = tf.gather(self.model.data['Xtrain'], indices)
        DAD = utils.calculate_SE_kernel(xs, self.model.ker_var, self.model.ker_length_scale,
                                        diag_noise=self.model.noise)
        normalizer = tf.sqrt(1.0 / tf.reduce_sum(DAD, axis=-1))
        normalizer = tf.matrix_diag(normalizer)
        DAD = tf.matmul(tf.matmul(normalizer, DAD), normalizer)
        # break symmetry
        masks = np.ones([self.model.hyper_param['n_parents'] + 1, self.model.hyper_param['n_parents'] + 1],
                        dtype=np.float64)
        masks[0, 1:] = 0
        masks = np.expand_dims(masks, axis=0)
        DAD = DAD * masks

        return ys, DAD

    def gather_L(self, indices):
        # gather ys and DAD
        ys, DAD = self.retrieve_ys_DAD(indices)
        # forward the neural network and retrieve the first row of the second dimension
        L = self.variational['L'].forward(ys, DAD)
        L_ii = tf.gather(L, [0], axis=-2)
        L_ii = tf.nn.softplus(L_ii)
        L_others = tf.gather(L, list(range(1, self.model.hyper_param['n_parents'] + 1)), axis=-2)
        L = tf.concat([L_ii, L_others], axis=-2)
        L = tf.squeeze(L, axis=-1)
        return L

    def gather_mu(self, indices):
        # gather ys and DAD
        ys, DAD = self.retrieve_ys_DAD(indices)
        # forward the neural network and retrieve the first row of the second dimension
        mu = self.variational['mu'].forward(ys, DAD)
        mu = tf.reduce_mean(mu, 1)
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
