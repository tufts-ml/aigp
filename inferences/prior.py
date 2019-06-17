import abc
from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import util.util as utils
from inferences.ent import Ent
from inferences.vanillaent import VanillaEnt


class GeneralPrior:
    def __init__(self, data, model_name='',
                 kernel_variance=1.0, length_scale=0.1, n_parents=20, batch_size=10, nn_conf=None,
                 opt_learning_rate=0.001, opt_max_iteration=100000000, exp_name='',
                 method='general', diag_noise=0.01):
        self.nn_conf = nn_conf
        self.method = method
        self.exp_num = exp_name
        # add a dummy node to data.
        random_X = data['Xtrain'][0:1] + 0.168 * np.ones([1, data['Xtrain'].shape[-1]])
        data['Xtrain'] = np.concatenate([random_X, data['Xtrain']], axis=0)
        data['ytrain'] = np.concatenate([data['ytrain'][0:1], data['ytrain']], axis=0)
        self.data = data
        self.n_Xtrain = data['Xtrain'].shape[0]
        # hyper parameters
        self.hyper_param = {'n_parents': n_parents}
        # initialize some constants
        self.cons = {'T': 500, 'batch_size': batch_size}
        parents_matrix, _, _ = utils.search_parents(self.data['Xtrain'], n_parents)
        self.cons['parents'] = tf.constant(parents_matrix, dtype=tf.int32)

        mask_sigma = np.ones([self.n_Xtrain, n_parents, n_parents], dtype=np.float64)
        for i in range(n_parents + 1):
            if i == 0 or i == 1:
                mask_sigma[i, :, :] = np.eye(n_parents)
                continue
            # upper right and lower left
            mask_sigma[i, :i - 1, i - 1:] = 0
            mask_sigma[i, i - 1:, :i - 1] = 0
            # lower right
            mask_sigma[i, i - 1:, i - 1:] = np.eye(n_parents - i + 1)
        self.mask_sigma = mask_sigma

        self.noise = diag_noise
        # kernel parameters
        self.ker_length_scale = tf.Variable(length_scale, dtype=tf.float64, trainable=False)
        self.ker_var = tf.Variable(kernel_variance, dtype=tf.float64, trainable=False)
        # initialize the inference function
        if self.method == 'general':
            self.variational_entropy = VanillaEnt(self)
        elif self.method == 'neural':
            self.variational_entropy = Ent(self)
        # initialize random stochastic variables
        self.single_rand = None
        # initialize optimize settings
        self.opt_setting = {'max_iteration': opt_max_iteration,
                            'param': opt_learning_rate,
                            'clip_gradients': True
                            }
        # save the model name
        self.model_name = model_name
        # open a session for Tensorflow
        self.sess = tf.Session()

        self.start_index = 1
        self.print_vars = []

    def gather_L(self, indices):
        if self.method == 'general':
            return tf.gather(self.variational_entropy.variational['L'], indices)
        elif self.method == 'neural':
            return self.variational_entropy.gather_L(indices)

    def gather_mu(self, indices):
        if self.method == 'general':
            return tf.gather(self.variational_entropy.variational['mu'], indices)
        elif self.method == 'neural':
            return self.variational_entropy.gather_mu(indices)

    def rand_point_stochastic_gradient(self):
        self.single_rand = tf.random_uniform([self.cons['batch_size']], minval=self.start_index, maxval=self.n_Xtrain,
                                             dtype=tf.int32)

    def calculate_sigma_cond_b(self, xs, indices):
        masks = tf.gather(self.mask_sigma, indices)
        # [B, 1, F]
        xs = tf.expand_dims(xs, axis=-2)
        # [B, K]
        parents = tf.gather(self.cons['parents'], indices)
        # [B, K, F]
        x_parents = tf.gather(self.data['Xtrain'], parents)
        # [B, B]
        sigma_ii = utils.calculate_SE_kernel(xs, self.ker_var, self.ker_length_scale, diag_noise=self.noise)
        # [B, 1]
        sigma_ii = tf.squeeze(sigma_ii, axis=-1)
        # [B, 1, K]
        sigma_ia = utils.calculate_SE_kernel(xs, self.ker_var, self.ker_length_scale, Y=x_parents)
        sigma_ia = sigma_ia * tf.to_double(tf.clip_by_value(tf.expand_dims(parents, axis=1), 0, 1))
        # [B, K, K]
        sigma_aa = utils.calculate_SE_kernel(x_parents, self.ker_var, self.ker_length_scale, diag_noise=self.noise)
        sigma_aa = sigma_aa * masks
        # [B, K, K]
        eye_matrix = tf.eye(self.hyper_param['n_parents'], batch_shape=[self.cons['batch_size']], dtype=tf.float64)
        # [B, K, K]
        chol = tf.cholesky(sigma_aa)
        sigma_aa_inv = tf.cholesky_solve(chol, eye_matrix)
        # [B, 1, K]
        b_vector = tf.matmul(sigma_ia, sigma_aa_inv)
        # [B, 1]
        sigma_cond = sigma_ii - tf.squeeze(tf.matmul(sigma_ia, tf.transpose(b_vector, perm=[0, 2, 1])), axis=-1)

        b_vector = tf.squeeze(b_vector)
        sigma_cond = tf.squeeze(sigma_cond)

        return b_vector, sigma_cond

    @abc.abstractmethod
    def cal_ell_term(self, indices):
        raise ValueError('ell term not implemented')

    def quad_difference_unique(self, batch_indices, parents, bi, sigma_ip):
        # calculate (Li - b_i^\top L_alpha(i)) sigma^-1_i/alpha(i) (Li - b_i^\top L_alpha(i))
        K = self.hyper_param['n_parents']
        batch_size = self.cons['batch_size']

        me_parents = tf.concat([tf.reshape(batch_indices, [batch_size, 1]), parents], axis=1)
        Li = self.gather_L(batch_indices)
        L_alpha = self.gather_L(tf.reshape(parents, [-1]))
        L_alpha = tf.reshape(L_alpha, [batch_size, K, K + 1])

        grand_parents = tf.gather(self.cons['parents'], parents)
        parent_grand = tf.concat([tf.expand_dims(parents, axis=2), grand_parents], axis=2)

        # first expand b
        b_hat = tf.concat([-tf.ones([batch_size, 1], dtype=tf.float64), bi], axis=1)

        # batch_size x (K + 1) x (K + 1)
        me_parent_grad = tf.concat([tf.expand_dims(me_parents, axis=1), parent_grand], axis=1)

        # batch_size x (K + 1) x (K + 1)
        L_hat = tf.concat([tf.expand_dims(Li, axis=1), L_alpha], axis=1)
        # mask the -1 parents (first K rows)
        L_hat = L_hat * tf.to_double(tf.clip_by_value(me_parent_grad, 0, 1))

        uni_n, uni_ind = tf.unique(tf.reshape(me_parent_grad, [-1]))
        me_parent_grad = tf.reshape(uni_ind, [batch_size, K + 1, K + 1])
        n_uni_ind = tf.size(uni_n)

        incremental_ind = tf.range(batch_size) * n_uni_ind

        column_ind = incremental_ind[:, None, None] + me_parent_grad
        # batch_size x (K + 1) x (K + 1)
        row_ind = tf.tile(tf.reshape(tf.range(K + 1), [1, (K + 1), 1]), [batch_size, 1, (K + 1)])
        # batch_size x (K + 1) x (K + 1)
        bL = tf.expand_dims(b_hat, axis=2) * L_hat

        sparse_bL = tf.SparseTensor(indices=tf.stack([tf.to_int64(tf.reshape(row_ind, [-1])),
                                                      tf.to_int64(tf.reshape(column_ind, [-1]))], axis=1),
                                    values=tf.reshape(bL, [-1]),
                                    dense_shape=[tf.to_int64(K + 1), tf.to_int64(batch_size * n_uni_ind)])

        # batch_size * n_unique vector
        dense_bL = tf.sparse_reduce_sum(sparse_bL, axis=0)
        dense_bL = tf.square(dense_bL)
        dense_bL = tf.reshape(dense_bL, [batch_size, n_uni_ind])
        dense_bL = tf.reduce_sum(dense_bL, axis=1) / sigma_ip
        dense_bL = tf.reshape(dense_bL, [batch_size])

        return dense_bL

    def cal_cross_term(self, batch_indices):
        # caution: might have index overflow problem when batch_size > 1000, N > 1 M

        # batch_size x K
        parents = tf.gather(self.cons['parents'], batch_indices)
        xs = tf.gather(self.data['Xtrain'], batch_indices)
        bi, sigma_ip = self.calculate_sigma_cond_b(xs, batch_indices)

        term1 = np.log(np.pi * 2) + tf.log(sigma_ip)

        term2 = self.quad_difference_unique(batch_indices, parents, bi, sigma_ip)
        # term2 = (1.0 + tf.reduce_sum(bi * bi, axis=1)) / sigma_ip
        # term2 = tf.square(1.0 - tf.reduce_sum(bi, axis=1)) / sigma_ip

        mu_i = tf.squeeze(self.gather_mu(batch_indices))
        mu_alpha = self.gather_mu(tf.reshape(parents, [-1]))
        mu_alpha = tf.reshape(mu_alpha, [tf.size(batch_indices), self.hyper_param['n_parents']])

        b_mu = tf.reduce_sum(mu_alpha * bi, axis=1)
        term3 = tf.square(mu_i - b_mu) / sigma_ip

        mean_cross = -0.5 * tf.reduce_mean(term1 + term2 + term3)

        return mean_cross

    def batch_ell(self, indices):
        ell = 0.0
        for itr in range(self.cons['batch_size']):
            ell += self.cal_ell_term(tf.gather(indices, [itr]))
        return ell / self.cons['batch_size']

    def cal_elbo(self):
        self.rand_point_stochastic_gradient()

        ell = self.batch_ell(self.single_rand)

        cross = self.cal_cross_term(self.single_rand)

        entropy = self.variational_entropy.cal_entropy_term(self.single_rand)

        return ell + cross + entropy

    def optimize(self):
        objectives = -self.cal_elbo()

        """ Using existing tf optimizer """
        opt = tf.train.AdamOptimizer(self.opt_setting['param']).minimize(objectives)

        self.sess.run(tf.global_variables_initializer())

        time_accumulator = 0.0
        num_itr = 0
        start = datetime.now()
        patience = 0
        best_min = np.infty

        print('Optimizing')
        for step in tqdm(range(1, self.opt_setting['max_iteration'] + 1)):
            self.sess.run([opt])

            if step % 100 == 0:
                time_accumulator += (datetime.now() - start).total_seconds()
                nlpd, lpd_arr, val_arr, _, _ = self.predict()

                if time_accumulator > 50000:
                    break
                cur_min = nlpd
                if cur_min + 0.01 < best_min:
                    best_min = cur_min
                    patience = 0
                else:
                    patience += 1
                if patience > 10:
                    break

                print('time: %.3f, test nll: %.3f' % (time_accumulator, -lpd_arr.mean()))

                start = datetime.now()

        return num_itr

    def predict_get_parents(self, x):
        dis = np.sum(np.square(self.data['Xtrain'] - x), axis=1)
        parents_indices = np.argsort(dis)[0:self.hyper_param['n_parents']]
        return np.squeeze(parents_indices)

    @abc.abstractmethod
    def predict_llp(self, q_samples, mu, sigma_square=0.25):
        raise ValueError('the way of drawing p(y|f) was not implemented')

    def predict(self, Xtest=None, Ytest=None):
        print('predicting...')
        if Xtest is None and Ytest is None:
            Xtest = self.data['Xtest']
            Ytest = self.data['ytest']
        total_lpd = 0
        lpd_arr = []
        val_arr = []
        n_parents = self.hyper_param['n_parents']
        n_xs, m_xs = Xtest.shape
        f_mu_ = []
        f_var_ = []

        # Tensorflow operations
        tf_x = tf.placeholder(dtype=tf.float64, shape=[1, m_xs])
        tf_parents = tf.placeholder(dtype=tf.int32, shape=[n_parents])
        tf_parents_n = tf.gather(self.cons['parents'], tf_parents)
        tf_mu_ns = self.gather_mu(tf_parents)
        tf_mu_ns = tf.reshape(tf_mu_ns, [-1])
        tf_L_n = self.gather_L(tf_parents)
        # calculate b vector
        x_parents = tf.gather(self.data['Xtrain'], tf_parents)
        tf_sigma_ii = utils.calculate_SE_kernel(tf_x, self.ker_var, self.ker_length_scale,
                                                diag_noise=self.noise)
        tf_sigma_ia = utils.calculate_SE_kernel(tf_x, self.ker_var, self.ker_length_scale,
                                                Y=x_parents, diag_noise=self.noise)
        tf_sigma_aa = utils.calculate_SE_kernel(x_parents, self.ker_var, self.ker_length_scale,
                                                diag_noise=self.noise)
        eye_matrix = tf.eye(self.hyper_param['n_parents'], dtype=tf.float64)
        sigma_aa_inv = tf.matrix_solve(tf_sigma_aa, eye_matrix)
        tf_b_vector = tf.matmul(tf_sigma_ia, sigma_aa_inv)
        # for prediction
        tf_pred_sample = tf.placeholder(dtype=tf.float64, shape=[1, self.cons['T']])
        tf_pred_mu = tf.placeholder(dtype=tf.float64)
        tf_pred_ipd, tf_pred_val = self.predict_llp(tf_pred_sample, tf_pred_mu)

        for i in range(n_xs):
            x = Xtest[i].reshape([1, m_xs])
            parents = utils.search_parents_predict(self.hyper_param['n_parents'], self.data['Xtrain'], x)
            parents = np.squeeze(parents)

            [mu_ns, L_n, parents_n, sigma_ii, sigma_ia, b_vector] = self.sess.run(
                [tf_mu_ns, tf_L_n, tf_parents_n, tf_sigma_ii, tf_sigma_ia, tf_b_vector],
                feed_dict={tf_x: x,
                           tf_parents: parents})

            i_and_parents = np.concatenate([np.expand_dims(parents, axis=1), parents_n], axis=1)
            u_nei, u_nei_indices = np.unique(i_and_parents, return_inverse=True)
            u_nei_indices = u_nei_indices.reshape([n_parents, (n_parents + 1)])
            L = np.zeros([n_parents, u_nei.shape[0]], dtype=np.float64)
            for j in range(n_parents):
                L[j].put(u_nei_indices[j], L_n[j])
            LL = np.matmul(L, L.T)

            E_f_s = np.matmul(b_vector, np.reshape(mu_ns, [-1, 1]))
            E_f_s = np.reshape(E_f_s, [1])

            var_f_s = sigma_ii - np.matmul(sigma_ia, b_vector.T) + np.matmul(np.matmul(b_vector, LL), b_vector.T)
            var_f_s = np.reshape(var_f_s, [1])

            # save for prediction
            f_mu_.append(E_f_s)
            f_var_.append(var_f_s)

            pred_samples = np.random.normal(E_f_s, np.sqrt(var_f_s), [1, self.cons['T']])
            pred_mu = Ytest[i]
            i_lpd, i_val = self.sess.run([tf_pred_ipd, tf_pred_val],
                                         feed_dict={tf_pred_sample: pred_samples, tf_pred_mu: pred_mu})

            total_lpd += i_lpd
            lpd_arr.append(i_lpd)
            val_arr.append(i_val)

        return -total_lpd / n_xs, np.array(lpd_arr), np.array(val_arr), np.array(f_mu_), np.array(f_var_)
