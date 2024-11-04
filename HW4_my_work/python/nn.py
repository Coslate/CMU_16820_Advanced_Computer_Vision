import numpy as np
from util import *

# do not include any more libraries here!
# do not put any code outside of functions!


############################## Q 2.1.2 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b.
# X be [Examples, Dimensions]

def randn(*shape, mean=0, std=1):
    # note: numpy doesn't support types within standard random routines, and
    # .astype("float32") does work if we're generating a singleton
    array = np.random.randn(*shape) * std + mean
    return array

def rand(*shape, low=0, high=1):
    # note: numpy doesn't support types within standard random routines, and
    # .astype("float32") does work if we're generating a singleton
    array = np.random.rand(*shape) * (high - low) + low
    return array 

def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain*(np.sqrt(2/(fan_in+fan_out)))
    shape = (fan_in, fan_out)
    return randn(*shape, mean=0, std=std)    

'''
def xavier_uniform2(fan_in, fan_out, gain=1.0, **kwargs):
    #np.random.seed(42) #for test
    a = gain*(np.sqrt(6/(fan_in+fan_out)))
    shape = (fan_in, fan_out)
    return rand(*shape, low=-a, high=a)
'''

def xavier_uniform(in_size,out_size):
    #np.random.seed(42) #for test
    a = np.sqrt(6/(in_size+out_size))
    return np.random.uniform(-a, a, (in_size, out_size))

def initialize_weights(in_size, out_size, params, name=""):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################
    W = xavier_uniform(in_size, out_size)
    b = np.zeros(out_size)

    params["W" + name] = W
    params["b" + name] = b


############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    res = 1/(1+np.exp(-x))

    return res


############################## Q 2.2.1 ##############################
def forward(X, params, name="", activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params["W" + name]
    b = params["b" + name]

    ##########################
    ##### your code here #####
    ##########################
    pre_act = X@W + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params["cache_" + name] = (X, pre_act, post_act)

    return post_act


############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    x_pron = x - np.max(x, axis=-1).reshape(-1, 1)
    exp_x = np.exp(x_pron)
    res = exp_x/np.sum(exp_x, axis=-1).reshape(-1, 1)
    return res


############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################
    loss = -np.sum(np.log(probs[np.arange(probs.shape[0]), np.argmax(y, axis=1)]))
    predictions = np.argmax(probs, axis=-1)
    true_classes = np.argmax(y, axis=-1)
    acc = np.mean(predictions == true_classes)

    return loss, acc


############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act * (1.0 - post_act)
    return res


def backwards(delta, params, name="", activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params["W" + name]
    b = params["b" + name]
    X, pre_act, post_act = params["cache_" + name]

    # do the derivative through activation first
    # (don't forget activation_deriv is a function of post_act)
    # then compute the derivative W, b, and X
    ##########################
    ##### your code here #####
    ##########################
    grad_Z = delta * activation_deriv(post_act)
    grad_X = grad_Z @ W.T
    grad_W = X.T @ grad_Z
    grad_b = np.sum(grad_Z, axis=0)

    # store the gradients
    params["grad_W" + name] = grad_W
    params["grad_b" + name] = grad_b
    return grad_X


############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def shuffle(X, y):
    """
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

def get_random_batches(x, y, batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################
    x, y = shuffle(x, y)
    for start_idx in range(0, len(y), batch_size):
        batch_x = x[start_idx:start_idx+batch_size, :]
        batch_y = y[start_idx:start_idx+batch_size, :]
        batches.append((batch_x, batch_y))

    return batches

#---------------Testing Execution---------------#
if __name__ == '__main__':
    params = {}
    w1 = xavier_uniform(3,5)
    #w2 = xavier_uniform2(3,5)

    print(f"w1 = {w1}")
    #print(f"w2 = {w2}")
