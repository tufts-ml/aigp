import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfd

from inferences.prior import GeneralPrior


class Poisson(GeneralPrior):
    def __init__(self, data, n_parents, variance, length_scale, method, model_name, nn_conf, opt_learning_rate,
                 opt_max_iteration=100):
        super(Poisson, self).__init__(data, model_name=model_name, n_parents=n_parents,
                                      kernel_variance=variance, length_scale=length_scale,
                                      method=method, nn_conf=nn_conf, opt_learning_rate=opt_learning_rate,
                                      opt_max_iteration=opt_max_iteration)

    def cal_ell_term(self, indices):
        y_point = tf.gather(self.data['ytrain'], indices)
        h_k = self.variational_entropy.sample(indices)
        z_k = tf.exp(h_k)

        dist = tfd.Poisson(rate=z_k)
        llh = tf.reduce_logsumexp(dist.log_prob(y_point)) - tf.log(tf.to_double(z_k.get_shape().as_list()[1]))

        return llh

    def predict_llp(self, q_samples, mu, ys=None):
        h_k = tf.nn.softplus(q_samples)

        dist = tfd.Poisson(rate=h_k)
        lpd = tf.reduce_logsumexp(dist.log_prob(mu)) - tf.log(tf.to_double(h_k.get_shape().as_list()[1]))

        return lpd, tf.to_double(0.0)


class Lognormal(GeneralPrior):
    def __init__(self, data, n_parents, variance, length_scale, method, model_name, nn_conf, opt_learning_rate,
                 opt_max_iteration=100):
        scale = 1.0
        self.free_scale = tf.Variable(scale, dtype=tf.float64)
        self.scale = tf.nn.softplus(self.free_scale)
        super(Lognormal, self).__init__(data, model_name=model_name, n_parents=n_parents,
                                        kernel_variance=variance, length_scale=length_scale,
                                        method=method, nn_conf=nn_conf, opt_learning_rate=opt_learning_rate,
                                        opt_max_iteration=opt_max_iteration)

    def log_p(self, f, y_true):
        sigma_square = tf.square(self.scale)
        llh = -0.5 * tf.log(2.0 * np.pi * sigma_square) - tf.log(y_true) - tf.square(tf.log(y_true)) / (
                    2.0 * sigma_square) + \
              tf.log(y_true) * f / sigma_square - tf.square(f) / (2.0 * sigma_square)
        log_prob = tf.reduce_logsumexp(llh) - tf.log(tf.to_double(f.get_shape().as_list()[1]))
        return log_prob

    def cal_ell_term(self, factor_index):
        y_point = tf.gather(self.data['ytrain'].astype(np.float64), factor_index)
        h_k = self.variational_entropy.sample(factor_index)
        points_llh = self.log_p(f=h_k, y_true=y_point)
        llh = tf.reduce_mean(points_llh)

        return llh

    def predict_llp(self, q_samples, mu, sigma_square=0.25):
        h_k = tf.identity(q_samples)
        lpd = self.log_p(f=h_k, y_true=mu)
        return lpd, tf.reduce_mean(h_k)


class Gaussian(GeneralPrior):
    def __init__(self, data, n_parents, variance, length_scale, method, model_name, nn_conf, opt_learning_rate,
                 opt_max_iteration=100):
        scale = 1.0
        self.free_scale = tf.Variable(scale, dtype=tf.float64)
        self.scale = tf.nn.softplus(self.free_scale)
        super(Gaussian, self).__init__(data, model_name=model_name, n_parents=n_parents,
                                       kernel_variance=variance, length_scale=length_scale,
                                       method=method, nn_conf=nn_conf, opt_learning_rate=opt_learning_rate,
                                       opt_max_iteration=opt_max_iteration)

    def log_p(self, f, y_true):
        dist = tfd.Normal(loc=f, scale=self.scale)
        llh = dist.log_prob(y_true)
        log_prob = tf.reduce_logsumexp(llh) - tf.log(tf.to_double(f.get_shape().as_list()[1]))
        return log_prob

    def cal_ell_term(self, index):
        y_point = tf.gather(self.data['ytrain'], index)
        h_k = self.variational_entropy.sample(index)
        point_llh = self.log_p(f=h_k, y_true=y_point)
        llh = tf.reduce_mean(point_llh)
        return llh

    def predict_llp(self, q_samples, mu, sigma_square=0.25):
        h_k = tf.identity(q_samples)
        lpd = self.log_p(f=h_k, y_true=mu)
        return lpd, tf.reduce_mean(h_k)

    def predict_vals(self, Xtest, Ytest):
        _, _, _, f_mu, f_var = self.predict(Xtest, Ytest)
        scale = self.sess.run(self.scale)
        return f_mu, f_var + scale ** 2


class Bernoulli(GeneralPrior):
    def __init__(self, data, n_parents, variance, length_scale, method, model_name, nn_conf, opt_learning_rate,
                 opt_max_iteration=100000000):
        super(Bernoulli, self).__init__(data, model_name=model_name, n_parents=n_parents,
                                        kernel_variance=variance, length_scale=length_scale,
                                        method=method, nn_conf=nn_conf, opt_learning_rate=opt_learning_rate,
                                        opt_max_iteration=opt_max_iteration)

    def log_p(self, f, y_true):
        llh = y_true * tf.log(f + 1e-6) + (1.0 - y_true) * tf.log(1.0 - f + 1e-6)
        log_prob = tf.reduce_logsumexp(llh) - tf.log(tf.to_double(f.get_shape().as_list()[1]))
        return log_prob

    def cal_ell_term(self, factor_index):
        y_point = tf.gather(self.data['ytrain'].astype(np.float64), factor_index)
        h_k = self.variational_entropy.sample(factor_index)
        h_k = tf.nn.sigmoid(h_k)
        points_llh = self.log_p(f=h_k, y_true=y_point)
        llh = tf.reduce_mean(points_llh)

        return llh

    def predict_llp(self, q_samples, mu, sigma_square=0.25):
        h_k = tf.nn.sigmoid(q_samples)
        lpd = self.log_p(f=h_k, y_true=mu)
        return lpd, tf.reduce_mean(h_k)
