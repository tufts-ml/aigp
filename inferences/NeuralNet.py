import tensorflow as tf


class NeuralNet:
    def __init__(self, input_dim, hidden_layer_dim, activate_func):
        """
        :param input_dim: scalar.
        :param hidden_layer_dim: list. Include the output layer.
        :param activate_func: list. Include the output layer.
        """
        self.hidden_layer_dim = hidden_layer_dim
        self.activate_func = activate_func

        # initialize the weights
        all_dims = [input_dim] + hidden_layer_dim
        self.W = [tf.Variable(tf.random_normal([all_dims[itr], all_dims[itr + 1]], stddev=0.1, dtype=tf.float64))
                  for itr in range(len(all_dims) - 1)]
        self.b = [tf.Variable(tf.random_normal([1, 1, all_dims[itr]], stddev=0.1, dtype=tf.float64))
                  for itr in range(1, len(all_dims))]

    def forward(self, X, DAD):
        n_hidden_layer = len(self.hidden_layer_dim)
        h = tf.identity(X)
        # forward the hidden layers
        for itr in range(n_hidden_layer - 1):
            h = tf.einsum('aij,jk->aik', tf.matmul(DAD, h), self.W[itr]) + self.b[itr]
            h = self.activate_func[itr](h)
        # the last layer
        h = tf.einsum('aij,jk->aik', tf.matmul(DAD, h), self.W[n_hidden_layer - 1]) + self.b[n_hidden_layer - 1]
        h = self.activate_func[n_hidden_layer - 1](h)
        return h
