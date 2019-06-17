import sys

sys.path.append('../')

from inferences.likelihood import *


def run(number_parents, variance, length_scale, opt_learning_rate, nn_conf, opt_max_iteration, dataset):
    # specify the dataset directory
    data_dir = 'your data directory'
    data = np.load(data_dir + 'data.npz')
    Xtrain, Ytrain, Xtest, Ytest = data['Xtrain'], data['Ytrain'], data['Xtest'], data['Ytest']
    data = {'Xtrain': Xtrain, 'ytrain': Ytrain, 'Xtest': Xtest, 'ytest': Ytest}

    # specify the likelihood
    likelihood = Poisson

    model = likelihood(data, n_parents=number_parents, variance=variance, length_scale=length_scale,
                       method='neural', model_name=dataset, opt_learning_rate=opt_learning_rate, nn_conf=nn_conf,
                       opt_max_iteration=opt_max_iteration)

    _ = model.optimize()

    return model

if __name__ == '__main__':
    # the length scale defined here is before the softplus transformation
    #           for example, length_scale=0.0 defined here really means length scale = softplus(0.0)=0.69

    nn_conf = {'mu_nn_conf': {'hidden_layer_dim': [20, 10, 1],
                              'activate_func': [tf.nn.relu, tf.nn.relu, tf.identity]},
               'L_nn_conf': {'hidden_layer_dim': [20, 10, 1],
                             'activate_func': [tf.nn.relu, tf.nn.relu, tf.identity]}}

    run(number_parents=10, variance=1.0, length_scale=0.1,
        opt_learning_rate=1e-2, nn_conf=nn_conf,
        opt_max_iteration=200, dataset='ebird')
