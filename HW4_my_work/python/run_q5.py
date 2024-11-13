import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36
#batch_size = 18
#batch_size = 96
learning_rate =  3e-5
#learning_rate =  3e-7
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################
# Initialize weights
n_inst, x_dimension = train_x.shape
n_inst_val, x_dimension_val = valid_x.shape
initialize_weights(x_dimension, hidden_size, params, "layer1")
initialize_weights(hidden_size, hidden_size, params, "layer2")
initialize_weights(hidden_size, hidden_size, params, "layer3")
initialize_weights(hidden_size, x_dimension, params, "output")
assert x_dimension == x_dimension_val

# Initialize momentum terms
upd_parameters_name = []
grad_parameters_name = []
momentum_params = Counter()
for key in params.keys():
    momentum_params[f'm_{key}'] = np.zeros_like(params[key])
    upd_parameters_name.append(key)
    grad_parameters_name.append('grad_'+key)


# should look like your previous training loops
losses = []
valid_losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   use 'm_'+name variables in initialize_weights from nn.py
        #   to keep a saved value
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################

        # forward pass
        h1_out = forward(xb    , params, 'layer1', relu)
        h2_out = forward(h1_out, params, 'layer2', relu)
        h3_out = forward(h2_out, params, 'layer3', relu)
        y_hat  = forward(h3_out, params, 'output', sigmoid)


        # loss
        total_loss += np.sum(np.power((y_hat - xb), 2))

        # backward
        grad_y_hat = 2*(y_hat -xb)
        delta3 = backwards(grad_y_hat, params, "output", sigmoid_deriv)
        delta2 = backwards(delta3    , params, "layer3", relu_deriv)
        delta1 = backwards(delta2    , params, "layer2", relu_deriv)
        _      = backwards(delta1    , params, "layer1", relu_deriv)

        # apply gradient, remember to update momentum as well
        for param_name, grad in zip(upd_parameters_name, [params[grad_parameters_name[i]] for i in range(len(grad_parameters_name))]):
            # Momentum update rule
            momentum_params[f'm_{param_name}'] = 0.9 * momentum_params[f'm_{param_name}'] - learning_rate * grad
            params[param_name] += momentum_params[f'm_{param_name}']

    '''
    h1_out_val = forward(valid_x   , params, 'layer1', relu)
    h2_out_val = forward(h1_out_val, params, 'layer2', relu)
    h3_out_val = forward(h2_out_val, params, 'layer3', relu)
    y_hat_val  = forward(h3_out_val, params, 'output', sigmoid)
    valid_loss = np.sum(np.power((y_hat_val - valid_x), 2))
    valid_losses.append(valid_loss / n_inst_val)
    '''
    
    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot loss curve
plt.plot(range(len(losses)), losses, 'r', label='Training Loss')  # Training loss in red
#plt.plot(range(len(valid_losses)), valid_losses, 'y', label='Validation Loss')  # Validation loss in blue
plt.xlabel("epoch")
plt.ylabel("average loss")
#plt.xlim(0, len(losses)-1)
plt.xlim(0, max(len(losses), len(valid_losses)) - 1)
plt.ylim(0, None)
plt.legend()  # Display legend to show color meaning
plt.grid()
plt.show()

        
# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["A", "B", "C", "1", "2"]

# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))
for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]

# run visualize_x through your network
# name the output reconstructed_x
##########################
##### your code here #####
##########################
# Forward Passing
h1_out = forward(visualize_x, params, 'layer1', relu)
h2_out = forward(h1_out     , params, 'layer2', relu)
h3_out = forward(h2_out     , params, 'layer3', relu)
reconstructed_x  = forward(h3_out     , params, 'output', sigmoid)

# visualize
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################
h1_out = forward(valid_x    , params, 'layer1', relu)
h2_out = forward(h1_out     , params, 'layer2', relu)
h3_out = forward(h2_out     , params, 'layer3', relu)
recon_val_x  = forward(h3_out     , params, 'output', sigmoid)

psnr_values = []
for original, reconstructed in zip(valid_x, recon_val_x):
    psnr = peak_signal_noise_ratio(original, reconstructed, data_range=original.max() - original.min())
    psnr_values.append(psnr)

# Calculate the average PSNR across all validation images
average_psnr = np.mean(psnr_values)
print(f"Average PSNR: {average_psnr}")
