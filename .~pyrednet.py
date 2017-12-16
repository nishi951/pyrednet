
import torch
import torch.cuda
import torch.optim as optim
# import torch.cuda as torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import os

import numpy as np

cuda = True

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight)
        nn.init.constant(m.bias, 0) 

class Prediction(nn.Module):
    """Prediction (Ahat) module
    Equals:
    SatLU(ReLU(Conv(input))) if layer = 0
    ReLU(Conv(input)) if layer > 0
    """
    def __init__(self, layer, inputChannels, outputChannels, filterSize,
                 pixel_max=1.):
        super(Prediction, self).__init__()
        self.layer = layer
        self.pixel_max = pixel_max
        self.conv = nn.Conv2d(inputChannels, outputChannels, filterSize, padding=(filterSize-1)//2)
        weights_init(self.conv)
        self.relu = nn.ReLU()
           
        
    def forward(self, x_in):
#         print("layer: {} prediction_in: {}".format(self.layer, x_in.shape))
        x = self.conv(x_in)
        x = self.relu(x)
        if self.layer == 0:
            x = torch.clamp(x, max=self.pixel_max)
#         print("layer: {}\n\tprediction_in: {}\n\tprediction_out: {}".format(self.layer, x_in.shape, x.shape))
        return x



#pred = Prediction(1, 3, 6, 3)
#weights_init(pred.conv)
#print(pred.conv.bias)


class Target(nn.Module):
    """Target (A) module
    Equals:
    input (x) if layer = 0
    MaxPool(ReLU(Conv(input))) if layer > 0 
    """
    def __init__(self, layer, inputChannels, outputChannels, filterSize):
        super(Target, self).__init__()
        self.layer = layer
        self.conv = nn.Conv2d(inputChannels, outputChannels, filterSize, padding=(filterSize-1)//2)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        weights_init(self.conv)
    
    def forward(self, x_in):
        x = self.conv(x_in)
        x = self.relu(x)
        x_out = self.maxpool(x)
#         print("layer: {}\n\ttarget_in: {}\n\ttarget_out: {}".format(self.layer, x_in.shape, x.shape))
        return x_out

class Error(nn.Module):
    """Error (E) module
    Input
    -----
    Images Ahat and A to be compared. Must have same number of channels.
    
    
    Output
    ------
    [ReLU(Ahat - A); ReLU(A - Ahat)]
    where concatenation is performed along the channel (feature) dimension
    aka the 0th dimension if A is of dimension.
    """
    def __init__(self, layer):
        super(Error, self).__init__()
        self.layer = layer
        self.relu = nn.ReLU()
        
    def forward(self, prediction, target):
#         print(target.shape)
#         print(prediction.shape)
        d1 = self.relu(target - prediction)
        d2 = self.relu(prediction - target)

        x = torch.cat((d1, d2), -3)
#         print("layer: {}\n\terror_in (x2): {}\n\terror_out: {}".format(self.layer, prediction.shape, x.shape))
        return x

# Representation Module Utility
class HardSigmoid(nn.Module):
    """
    Re-implementation of HardSigmoid from Theano. (Rolfo, Nishimura 2017)
    
    Constrained to be 0 when x <= min_val and
    1 when x >= max_val, 
    and to be linear in between the two ranges.
    """
    def __init__(self, min_val, max_val):
        super(HardSigmoid, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.hardtanh = nn.Hardtanh(min_val, max_val)
    def forward(self, x_in):
        x = self.hardtanh(x_in)
        x = (x - self.min_val)/(self.max_val - self.min_val)
        return x

# m = (nn.Hardtanh(-2.5,2.5) + torch.Tensor(2.5))/5
# m = HardSigmoid(-2.5, 2.5)
# x = torch.autograd.Variable(torch.randn(2))
# print(x)
# print(m(x))


class Representation(nn.Module):
    """Representation (R) module
    
    A ConvLSTM (https://arxiv.org/pdf/1506.04214.pdf) unit.
    
    Actually, it's implemented differently in the prednet paper,
    so we decided to mimic that implementation as much as possible.
    """
    def __init__(self, layer, numLayers, 
                 R_stack_sizes, A_stack_sizes,
                 kernel_size,
                 c_width_height=None,
                 peephole=False, hardsigmoid_min=-2.5, hardsigmoid_max=2.5):
        
        # keras: Conv2D(filters aka out_channels, kernel_size, strides=(1, 1), padding='valid', data_format=None
        # KERAS: Conv2D(self.R_stack_sizes[l], self.R_filt_sizes[l], padding='same', activation=act, data_format=self.data_format))
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        
        # in_channels -> Lotter. should be self.R_stack_sizes[l] + e_stack_sizes[l] + if(not last) R_stack_sizes[l+1]
        # out_channels -> Lotter self.R_stack_sizes[l], 
        # kernel_size -> Lotter self.R_filt_sizes[l]. all 3s
        super(Representation, self).__init__()
        self.layer = layer
        self.numLayers = numLayers
        self.peephole = peephole
        
        ### Keep track of sizes of inputs ###
        inputChannels = {}
        self.E_stack_sizes = tuple(2*stack_size for stack_size in A_stack_sizes)
        self.C_stack_size = R_stack_sizes[layer] # the cell size for this layer
        if (layer == numLayers-1):
            for gate in ['i', 'f', 'o', 'c']:
                inputChannels[gate] = R_stack_sizes[layer] + self.E_stack_sizes[layer]
        else:
            for gate in ['i', 'f', 'o', 'c']:
                inputChannels[gate] = R_stack_sizes[layer] + self.E_stack_sizes[layer] + R_stack_sizes[layer+1]
        ### END ###
            
        if peephole:
#             self.Wc = {} # Parameters for hadamard (elementwise) products
#             for gate in ['i', 'f', 'o']:
#                 self.Wc[gate] = torch.Parameter(torch.FloatTensor(inputChannels['c'], c_width_height[0], c_width_height[1]))
            self.Wc_i = torch.Parameter(torch.FloatTensor(inputChannels['c'], c_width_height[0], c_width_height[1]))
            self.Wc_f = torch.Parameter(torch.FloatTensor(inputChannels['c'], c_width_height[0], c_width_height[1]))
            self.Wc_o = torch.Parameter(torch.FloatTensor(inputChannels['c'], c_width_height[0], c_width_height[1]))
        
#         self.conv = {}
#         self.act = {}
        outputChannels = R_stack_sizes[layer]
#         for gate in ['i', 'f', 'o', 'c']:
#             print((kernel_size-1)/2)
#             self.conv[gate] = nn.Conv2d(inputChannels[gate], outputChannels, kernel_size, padding=(kernel_size-1)//2)

        self.conv_i = nn.Conv2d(inputChannels[gate], outputChannels, kernel_size, padding=(kernel_size-1)//2)
        self.conv_f = nn.Conv2d(inputChannels[gate], outputChannels, kernel_size, padding=(kernel_size-1)//2)
        self.conv_o = nn.Conv2d(inputChannels[gate], outputChannels, kernel_size, padding=(kernel_size-1)//2)
        self.conv_c = nn.Conv2d(inputChannels[gate], outputChannels, kernel_size, padding=(kernel_size-1)//2)

        weights_init(self.conv_i)
        weights_init(self.conv_f)
        weights_init(self.conv_o)
        weights_init(self.conv_c)
        
#         self.act['i'] = HardSigmoid(hardsigmoid_min, hardsigmoid_max)
#         self.act['f'] = HardSigmoid(hardsigmoid_min, hardsigmoid_max)
#         self.act['o'] = HardSigmoid(hardsigmoid_min, hardsigmoid_max)
#         self.act['c'] = nn.Tanh()
#         self.act['h'] = nn.Tanh()
        self.act_i = HardSigmoid(hardsigmoid_min, hardsigmoid_max)
        self.act_f = HardSigmoid(hardsigmoid_min, hardsigmoid_max)
        self.act_o = HardSigmoid(hardsigmoid_min, hardsigmoid_max)
        self.act_c = nn.Tanh()
        self.act_h = nn.Tanh()
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        
        
    def forward(self, e_prev, r_prev, c_prev, r_above):
        """
        e_prev: the error input at (l, t-1)
        r_prev: the input to ahat of the representation cell at (l, t-1)
        c_prev: the cell (internal) of the representation cell at (l, t-1)
        r_above: the input to ahat of the rep.cell at (l+1, t) -- to ignore if l = L
        
        # Lotter implementation, which is bastardized CLSTM
        """

        stacked_inputs = torch.cat((r_prev, e_prev), -3)
        if (self.layer < self.numLayers-1):
            r_above_up = self.upsample(r_above)
            stacked_inputs = torch.cat((stacked_inputs, r_above_up), -3)
        
        # Calculate hidden cell update:
        # First get gate updates
#         gates = {}
#         for gate in ['i', 'f', 'o']:
#             gates[gate] = self.conv[gate](stacked_inputs)
#         if self.peephole:
#             gates['f'] += torch.mul(self.Wc['f'], c_prev)
#             gates['i'] += torch.mul(self.Wc['i'], c_prev)

        # Compute gates
        i = self.conv_i(stacked_inputs)
        f = self.conv_f(stacked_inputs)
        o = self.conv_o(stacked_inputs)
        if self.peephole:
            f = f + self.Wc_f * c_prev
            i = i + self.Wc_i * c_prev
        i = self.act_i(i)
        f = self.act_f(f)
            


        # Update hidden cell
#         print('gates f', gates['f'].shape)
#         print('gates i', gates['i'].shape)
#         print('gates o', gates['o'].shape)
#         print('stacked inputs', stacked_inputs.shape)
#         print('cprev', c_prev.shape)
#         print(c_prev.shape)
#         print(stacked_inputs.shape)
#         print(gates['f'].shape)
#         print("layer: {}\n\tr_in: {}\n\tr_out: {} \
#               \n\tc_in: {}\n\tc_out: {}\n\terror".format(self.layer, 
#                                                                         r_prev.shape, r.shape,
#                                                                         c_prev.shape, c.shape,
#                                                                         e_prev.shape))
#         c = gates['f'] * c_prev + gates['i'] * \
#             self.act['c'](self.conv['c'](stacked_inputs))

        # Update hidden cell
        c = f * c_prev + i * self.act_c(self.conv_c(stacked_inputs))
    
        # o gate uses current cell, not previous cell
        if self.peephole:
            o = o + self.Wc_o * c
        o = self.act_o(o)


#         r = torch.mul(gates['o'], self.act['c'](c))
        r = o * self.act_h(c)
#         print(r.shape)
        
#         print("layer: {}\n\tr_in: {}\n\tr_out: {} \
#               \n\tc_in: {}\n\tc_out: {}\n\terror".format(self.layer, 
#                                                                         r_prev.shape, r.shape,
#                                                                         c_prev.shape, c.shape,
#                                                                         e_prev.shape))
        return r, c

class PredNet(nn.Module):
    """Full stack of layers in the prednet architecture
    |targets|, |errors|, |predictions|, and |representations|
    are all lists of nn.Modules as defined in their respective class
    definitions.
    
    If the prednet has L prednet layers, then
    targets has length L-1
    errors, predictions, and representations have length L
    
    """
    def __init__(self, targets, errors, 
                 predictions, representations,
                 numLayers, R_stack_sizes, stack_sizes, heights, widths):
        super(PredNet, self).__init__()
        self.targets = nn.ModuleList(targets)
        self.errors = nn.ModuleList(errors)
        self.predictions = nn.ModuleList(predictions)
        self.representations = nn.ModuleList(representations)
        
        assert targets[0] is None # First target layer is just the input
    
        self.numLayers = numLayers
        self.R_stack_sizes = R_stack_sizes
        self.stack_sizes = stack_sizes
        self.heights = heights
        self.widths = widths
    
    def forward(self, x_in, r_prev, c_prev, e_prev):
        """
        Arguments:
        ----------
        |r_prev| and |c_prev| are lists of previous values of 
        the outputs r and hidden cells c of the representation cells
        
        |e_prev| is the list of errors for the last iteration

        |x_in| is the next minibatch of inputs, of size
        (num_examples, num_channels, width, height)
        """
        r = [None for _ in range(self.numLayers)]
        c = [None for _ in range(self.numLayers)]
        e = [None for _ in range(self.numLayers)]

        # First, update all the representation layers:
        # Do the top layer first
        last = self.numLayers - 1
        r[last], c[last] = \
            self.representations[last](e_prev[last], r_prev[last],
                                        c_prev[last], None)

        for layer in range(last-1, -1, -1):
            r[layer], c[layer] = self.representations[layer](e_prev[layer],
                                                             r_prev[layer],
                                                             c_prev[layer],
                                                             r_prev[layer+1])
        # Bottom layer gets input instead of a target cell
        prediction = self.predictions[0](r[0])
        e[0] = self.errors[0](prediction, x_in)
        # layer 1 through layer numLayers-1
        for layer in range(1, self.numLayers):
            target = self.targets[layer](e[layer-1])
            prediction = self.predictions[layer](r[layer])
            e[layer] = self.errors[layer](prediction, target)
        return r, c, e
    
    def init_representations(self):
        """
        Return the initial states of the representations, the cells, and the errors.
        """
        R_init = [Variable(torch.zeros(self.R_stack_sizes[layer], self.heights[layer], self.widths[layer])).cuda() 
                  for layer in range(len(self.R_stack_sizes))]
        E_init = [Variable(torch.zeros(2*self.stack_sizes[layer], self.heights[layer], self.widths[layer])).cuda()
                  for layer in range(len(self.stack_sizes))]
        C_init = [Variable(torch.zeros(self.R_stack_sizes[layer], self.heights[layer], self.widths[layer])).cuda()
                  for layer in range(len(self.stack_sizes))]

        return R_init, C_init, E_init

# convlayer = nn.Conv2d(3, 6, 3)
# data = Variable(torch.randn(4, 3, 5, 5))
# data2 = Variable(torch.randn(4, 3, 5, 5))
# out = torch.cat((data, data2), -3)
# print(out.shape)
# out2 = out.expand(4, -1, -1, -1, -1)
# print(out2.shape)


### Loss Function ###
def PredNetLoss(Errors, layer_weights, time_weights, batch_size):
    """
    Computes the weighted L1 Loss over time and over all the layers
    Parameters
    ----------
    Errors - list of lists of error tensors E, where
             Errors[i][j] is the error tensor of the ith time step
             at the jth prednet layer.
             
    time_weights - weights that govern how much each time step contributes
                   to the overall loss.
                   
    layer_weights - lambdas that govern how much each prednet layer error
                    contributes to the overall loss

    """    
    overallLoss = Variable(torch.zeros(1)).cuda()
#     overallLoss = Variable(torch.cuda.zeros(1))
    for i, t_weight in enumerate(time_weights):
        timeLoss = Variable(torch.zeros(1)).cuda()
#         timeLoss = Variable(torch.cuda.zeros(1))
        for j, l_weight in enumerate(layer_weights):
            E = Errors[i][j]
            timeLoss = timeLoss + l_weight*torch.mean(E)
#         print(type(t_weight))
#         print(type(timeLoss))
#         print(type(overallLoss))
        overallLoss = overallLoss + t_weight * timeLoss
    overallLoss = overallLoss/batch_size
    return overallLoss

### Training procedure ###
def train(train_data, num_epochs, epoch_size, batch_size, optimizer, 
          prednet, loss_weights):
    """
    Parameters
    ----------
    train_data - Iterator that produces input tensors of size
                 batch_size x time x channels x width x height
                 suitable for input into the network.
                  
    num_epochs - number of passes over the training data
    optimizer - torch.optim object
    prednet - model, for forward and backward calls
    loss_weights - tuple of (layer_loss_weights, time_loss_weights) - parameters
                   for computing the loss
    """

    # Initialize the optimizer.
    optimizer.zero_grad()

    losses = []
    print("Training...")
    # Iterate through the time dimension
    for epoch in range(num_epochs):
        print("Epoch {}".format(epoch))
        epochLoss = []
        for it, data in enumerate(train_data):
            if it == epoch_size: # Don't pass over entire training set.
                break
            data = Variable(data)
            batch_size, timesteps, channels, width, height = data.shape

            r, c, e = prednet.init_representations()
            r = [rep.expand(batch_size, -1, -1, -1) for rep in r]
            c = [cell.expand(batch_size, -1, -1, -1) for cell in c]
            e = [err.expand(batch_size, -1, -1, -1) for err in e]
            errorCells = []            
            for t in range(timesteps):
                r, c, e = prednet(data[:,t,:,:,:], r, c, e)
                errorCells.append(e)
            # Compute the loss (custom):
            loss = PredNetLoss(errorCells, layer_loss_weights, time_loss_weights, batch_size)
            loss.backward()

            # Add parameters' gradients to their values, multiplied by learning rate    
            optimizer.step()
            # Save loss somehow
            epochLoss.append(loss.data[0])
            print("\tIteration: {} loss: {}".format(it, loss.data[0]))
        losses.append(epochLoss)
        
        
        
    return losses

save_model = True  # if weights will be saved
# weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')  # where weights will be saved
# json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')

# Data files
# train_file = os.path.join(DATA_DIR, 'X_train.hkl')
# train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
# val_file = os.path.join(DATA_DIR, 'X_val.hkl')
# val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')


# Model parameters
numLayers = 4
pixel_max = 1.0
nt = 10 # Number of frames in each training example video
n_channels, im_height, im_width = (3, 128, 160)
widths = [im_width//(2**layer) for layer in range(numLayers)]
heights = [im_height//(2**layer) for layer in range(numLayers)]
# c_width_height = Only necessary if peephole=true
input_shape = (n_channels, im_height, im_width) #if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
stack_sizes = (n_channels, 48, 96, 192)
E_stack_sizes = tuple(2*stack_size for stack_size in stack_sizes)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
# layer_loss_weights = Variable(torch.FloatTensor(np.array([1., 0., 0., 0.])),
#                               requires_grad=False)# weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = Variable(torch.cuda.FloatTensor(np.array([1., 0., 0., 0.])),
                              requires_grad=False)# weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]


#layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
nt = 10  # number of timesteps used for sequences in training
# time_loss_weights = Variable(torch.FloatTensor(1./ (nt - 1) * np.ones((nt,1))),
#                              requires_grad=False) # equally weight all timesteps except the first
time_loss_weights = Variable(torch.cuda.FloatTensor(1./ (nt - 1) * np.ones((nt,1))),
                             requires_grad=False) # equally weight all timesteps except the first
time_loss_weights[0] = 0
loss_weights = (layer_loss_weights, time_loss_weights)

### Initialize the network ###
targets = [None for _ in range(numLayers)]
predictions = [None for _ in range(numLayers)]
representations = [None for _ in range(numLayers)]
errors = [None for _ in range(numLayers)]
for layer in range(numLayers):
    predictions[layer] = Prediction(layer,
                                  R_stack_sizes[layer],
                                  stack_sizes[layer],
                                  Ahat_filt_sizes[layer])
#     print(R_filt_sizes[layer])


    representations[layer] = Representation(layer,
                                          numLayers,
                                          R_stack_sizes,
                                          stack_sizes,
                                          R_filt_sizes[layer]
                                         )
                          
    errors[layer] = Error(layer)
    if layer > 0:
        targets[layer] = Target(layer,
                                2*stack_sizes[layer-1],
                                stack_sizes[layer],
                                A_filt_sizes[layer-1]
                               )

prednet = PredNet(targets, errors, predictions, representations,
                  numLayers, R_stack_sizes, stack_sizes, heights, widths)
prednet.cuda() # Comment out for cpu

### Load the dataset ###
# from Lotter: train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)

DATA_DIR = './kitti_data/'
#train_file = os.path.join(DATA_DIR, 'X_train.hkl')
#train_sources = os.path.join(DATA_DIR, 'sources_train.p')
train_file = './kitti_data/X_train.p'
train_source = './kitti_data/sources_train.p'
val_file = './kitti_data/X_val.p'
val_source = './kitti_data/sources_val.p'
nt = 10  # number of timesteps used for sequences in training

class KittiDataset(Dataset): 
    def __init__(self, data_file, source_file, nt, transform=None, output_mode='error', shuffle=False):
        
        """
        Args:
            data_file (string): Path to the hickle file with dimensions (n_imgs, height, width, num channels)
                                                        for train_file: (41396, 128, 160, 3)
            source_file (string): hickle file of list with all the images, with length n_imgs
            transform (callable, optional): Optional transform to be applied
                on a sample.
            nt: number of timesteps for sequences in training
            # do we need to consider channels first/last?
        """
#         self.X = torch.FloatTensor(pkl.load(open(data_file, 'rb')).astype(np.float32)/255)
        self.X = torch.cuda.FloatTensor(pkl.load(open(data_file, 'rb')).astype(np.float32)/255)
        self.sources = pkl.load(open(source_file, 'rb'))
        self.nt = nt
        
        self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        
        print(len(self.possible_starts))
        
    def __len__(self):
        return len(self.possible_starts)

    
    # DEFINE AN EXAMPLE AS NT TIMESTEPS OF A SEQUENCE
    def __getitem__(self, idx):
        data = self.X[idx:idx+nt,:,:,:]
        # want dimensions to be (time, channels, width, height)
        data = torch.transpose(data, 1, 3)
        data = torch.transpose(data, 2, 3)
        return data

# 
# dataset = KittiDataset(train_file_pkl, train_source_pkl, nt)
dataset = KittiDataset(val_file, val_source, nt)



### Train the network ###

# Replace this with something like training with adam
# Training parameters
num_epochs = 100
batch_size = 4
epoch_size = 500
N_seq_val = 100  # number of sequences to use for validation
learning_rate = 1e-3
print('lr', learning_rate)

optimizer = optim.Adam(prednet.parameters(), lr=learning_rate)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# pin_memory=True
losses = train(loader, num_epochs, epoch_size, batch_size, optimizer, 
          prednet, loss_weights)



import matplotlib.pyplot as plt
flat_losses = [item for sublist in losses for item in sublist]

plt.plot(np.log(flat_losses))
plt.title("Training loss")
plt.xlabel("iteration")
plt.ylabel("log(Loss)")
plt.show()
