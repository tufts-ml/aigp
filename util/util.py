import numpy as np
import tensorflow as tf
from tqdm import tqdm


def search_parents_predict(n_neighbors, X, Y):
    distances = np.sum(np.square(X - Y), axis=1)
    neighbors = np.argsort(distances)[:n_neighbors].astype(np.int32)
    return neighbors


def calculate_SE_kernel(X, kernel_variance, kernel_length_scale, Y=None, diag_noise=1e-3):
    # use softplus to replace square
    squared_kernel_variance = (kernel_variance)
    squared_kernel_length_scale = (kernel_length_scale)
    # check if it is used to calculate prior covariance
    if Y is None:
        Y = tf.identity(X)
        diag_noise = tf.diag(diag_noise * tf.ones(X.get_shape().as_list()[-2], dtype=tf.float64))
        if len(X.get_shape().as_list()) > 2:
            diag_noise = tf.expand_dims(diag_noise, axis=0)
    else:
        diag_noise = 0.0
    # expand dimension for X and Y
    X = tf.expand_dims(X, axis=-2)
    Y = tf.expand_dims(Y, axis=-3)
    # calculate distance
    distance = tf.reduce_sum(tf.square(X - Y), axis=-1)
    # calculate kernel matrix
    kernel_matrix = squared_kernel_variance * tf.exp(-0.5 * distance / squared_kernel_length_scale) + diag_noise

    return kernel_matrix


def gaussian_1d_log_likelihood(z, mu, sigma=0.1):
    dist = tf.distributions.Normal(loc=tf.reshape(z, [-1]), scale=tf.to_double(sigma))
    log_probs = dist.log_prob(mu)
    llh = tf.reduce_logsumexp(log_probs) - tf.log(tf.to_double(z.get_shape().as_list()[1]))
    return llh


def poisson_1d_log_likelihood(z, mu):
    dist = tf.contrib.distributions.Poisson(rate=z)
    llh = tf.reduce_logsumexp(dist.log_prob(mu)) - tf.log(tf.to_double(z.get_shape().as_list()[1]))
    return llh


def lognormal_1d_log_likelihood(z, mu, sigma_square=0.01):
    sigma_square = tf.to_double(sigma_square)
    each_lpd = -0.5 * tf.log(2 * np.pi * sigma_square) - tf.log(mu) - tf.log(mu) ** 2 / (2 * sigma_square) + \
               tf.log(mu) * z / sigma_square - tf.square(z) / (2 * sigma_square)
    llh = tf.reduce_logsumexp(each_lpd) - tf.log(tf.to_double(z.get_shape().as_list()[1]))

    return llh


def exp_1d_log_likelihood(z, mu):
    llh = tf.reduce_logsumexp(-mu / z - tf.log(z))
    return llh


def beta_1d_log_likelihood(z, mu, beta=2.0):
    # need to clip y, since log of 0 is nan...
    y = tf.clip_by_value(mu, 1e-6, 1 - 1e-6)
    dist = tf.distributions.Beta(mu, beta)
    llh = tf.reduce_logsumexp(dist.log_prob(y)) - tf.log(tf.to_double(z.get_shape().as_list()[1]))
    return llh


def gamma_1d_log_likelihood(z, mu, shape=10.0):
    each_llh = -shape * tf.log(z) - tf.lgamma(shape) + (shape - 1.) * tf.log(mu) - mu / z
    llh = tf.reduce_logsumexp(each_llh, axis=1)
    return llh


def search_parents(locations, n_neighbor):
    n, m = locations.shape
    # we will force the first parent to be itself.
    parents = np.zeros([n, n_neighbor], dtype=np.int32)
    # the current index for each parent to be filled. Start from 1, since the first parent are themselves.
    parents_indices = np.zeros(n, dtype=np.int32)

    print('Building DAG...')
    for i in tqdm(range(1, n)):
        # calculate the first K nearest neighbors
        distances = np.sum(np.square(locations[i] - locations), axis=1)
        neighbors = np.argsort(distances)

        # fill up the parents for i

        j = 0
        while parents_indices[i] < min(i, n_neighbor) and j < n:
            neighbor_id = neighbors[j]
            if neighbor_id < i and np.sum(parents[i] == neighbor_id) == 0:
                parents[i, parents_indices[i]] = neighbor_id
                parents_indices[i] += 1
            j += 1

        # fill up the children for i
        for j in range(n_neighbor):
            neighbor_id = neighbors[j]
            if neighbor_id > i and parents_indices[neighbor_id] < min(neighbor_id, n_neighbor) and \
                    np.sum(parents[neighbor_id] == i) == 0:
                parents[neighbor_id, parents_indices[neighbor_id]] = i
                parents_indices[neighbor_id] += 1

    return parents, 0, 0
