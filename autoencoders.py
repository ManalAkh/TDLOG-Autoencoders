import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split as ttsplit

class SimpleAutoEncoder(nn.Module):
    def __init__(self, input_dim,bottleneck_dim):
        """Initialise simplest autoencoder (input->bottleneck->ouput), with hyperbolic tangent activation function
       
        :param input_dim: int, Number of dimension of the input vectors
        :param bottleneck_dim: int, Number of dimension of the bottleneck
        """
        super(SimpleAutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, bottleneck_dim),
            torch.nn.Tanh()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_dim, input_dim),
        )

    def forward(self, inp):
        encoded = self.encoder(inp)
        decoded = self.decoder(encoded)
        return decoded
    
class DeepAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, bottleneck_dim):
        """Initialise auto encoder with hyperbolic tangent activation function
        You can uncomment certain lines in the encoder and decoder functions to modify the topology of the network
        Make sure when you initialise the AE object that the list 'hidden_dims' has a length consistent with the architecture

        :param input_dim: int, Number of dimension of the input vectors
        :param hidden_dims: list, List of hidden layers
        :param bottleneck_dim: int, Number of dimension of the bottleneck
        """
        super(DeepAutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dims[0]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dims[1], hidden_dims[2]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dims[-1], bottleneck_dim),
            torch.nn.Tanh()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_dim, hidden_dims[-1]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dims[-1], hidden_dims[-2]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dims[-2], hidden_dims[-3]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dims[0], input_dim),
        )

    def forward(self, inp):
        # Input Linear function
        encoded = self.encoder(inp)
        decoded = self.decoder(encoded)
        return decoded

def set_learning_parameters(model, learning_rate, loss='MSE', optimizer='Adam'):
    """Function to set learning parameter

    :param model: Neural network model build with PyTorch,
    :param learning_rate: Value of the learning rate
    :param loss: String, type of loss desired ('MSE' by default, another choice leads to cross entropy)
    :param optimizer: String, type of optimizer ('Adam' by default, another choice leads to SGD)

    :return:
    """
    #--- chosen loss function ---
    if loss == 'MSE':
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.CrossEntropyLoss()
    #--- chosen optimizer ---
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return loss_function, optimizer

def train_AE(model, loss_function, optimizer, traj, weights, num_epochs=10, batch_size=32, test_size=0.2):
    """Function to train an AE model

    :param model: Neural network model built with PyTorch,
    :param loss_function: Function built with PyTorch tensors or built-in PyTorch loss function
    :param optimizer: PyTorch optimizer object
    :param traj: np.array, physical trajectory (in the potential pot), ndim == 2, shape == T // save + 1, pot.dim
    :param weights: np.array, weights of each point of the trajectory when the dynamics is biased, ndim == 1, shape == T // save + 1, 1
    :param num_epochs: int, number of times the training goes through the whole dataset
    :param batch_size: int, number of data points per batch for estimation of the gradient
    :param test_size: float, between 0 and 1, giving the proportion of points used to compute test loss

    :return: model, trained neural net model
    :return: training_data, list of lists of train losses and test losses; one per batch per epoch
    """
    #--- prepare the data ---
    # split the dataset into a training set (and its associated weights) and a test set
    X_train, X_test, w_train, w_test = ttsplit(traj, weights, test_size=test_size)
    X_train = torch.tensor(X_train.astype('float32'))
    X_test = torch.tensor(X_test.astype('float32'))
    w_train = torch.tensor(w_train.astype('float32'))
    w_test = torch.tensor(w_test.astype('float32'))
    # intialization of the methods to sample with replacement from the data points (needed since weights are present)
    train_sampler = torch.utils.data.WeightedRandomSampler(w_train, len(w_train))
    test_sampler  = torch.utils.data.WeightedRandomSampler(w_test, len(w_test))
    # method to construct data batches and iterate over them
    train_loader = torch.utils.data.DataLoader(dataset=X_train,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               sampler=train_sampler)
    test_loader  = torch.utils.data.DataLoader(dataset=X_test,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               sampler=test_sampler)
    
    #--- start the training over the required number of epochs ---
    training_data = []
    for epoch in range(num_epochs):
        # Train the model by going through the whole dataset
        model.train()
        train_loss = []
        for iteration, X in enumerate(train_loader):
            # Set gradient calculation capabilities
            X.requires_grad_()
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output
            out = model(X)
            # Evaluate loss
            loss = loss_function(out, X)
            # Store loss
            train_loss.append(loss)
            # Get gradient with respect to parameters of the model
            loss.backward()
            # Updating parameters
            optimizer.step()
        # Evaluate the test loss on the test dataset
        model.eval()
        with torch.no_grad():
            test_loss = []
            for iteration, X in enumerate(test_loader):
                out = model(X)
                # Evaluate loss
                loss = loss_function(out, X)
                # Store loss
                test_loss.append(loss)
            training_data.append([torch.tensor(train_loss), torch.tensor(test_loss)])
    return model, training_data

def xi_ae(model,  x):
    """Collective variable defined through an auto encoder model

    :param model: Neural network model build with PyTorch
    :param x: np.array, position, ndim = 2, shape = (1,1)

    :return: xi: np.array
    """
    model.eval()
    x = torch.tensor(x.astype('float32'))
    return model.encoder(x).detach().numpy()

def grad_xi_ae(model, x):
    """Gradient of the collective variable defined through an auto encoder model

    :param model: Neural network model build with pytorch,
    :param x: np.array, position, ndim = 2, shape = (1,1)

    :return: grad_xi: np.array
    """
    model.eval()
    x = torch.tensor(x.astype('float32'))
    x.requires_grad_()
    enc = model.encoder(x)
    grad = torch.autograd.grad(enc, x)[0]
    return grad.detach().numpy()

def plot_results(potential, trajectory):
    """
        :param potential:  MultimodalPotential, potential
        :param trajectory: np.array

        :return: Fig1: Figure, training and test losses plots
        :return: Fig2: Figure, Collective variables plots
        """
    learning_rate = 0.005
    batch_size = 100
    num_epochs = 20
    loss = 'MSE'
    optimizer = 'Adam'
    ae0 = SimpleAutoEncoder(2,1) 
    ae1 = DeepAutoEncoder(2, [4,5,3], 1)
    loss_function, optimizer = set_learning_parameters(ae0, learning_rate=learning_rate, loss=loss, optimizer=optimizer)
    ae0, training_data0 = train_AE(ae0,
                                loss_function,
                                optimizer,
                                trajectory,
                                np.ones(trajectory.shape[0]),
                                batch_size=batch_size,
                                num_epochs=num_epochs
                                )
    loss_function, optimizer = set_learning_parameters(ae1, learning_rate=learning_rate, loss=loss, optimizer=optimizer)
    ae1, training_data1 = train_AE(ae1,
                                loss_function,
                                optimizer,
                                trajectory,
                                np.ones(trajectory.shape[0]),
                                batch_size=batch_size,
                                num_epochs=num_epochs
                                )

    #--- plot the evolution of the losses ---
    loss_evol0 = []
    loss_evol1 = []
    # obtain average losses on each epoch by averaging the losses from each batch
    for i in range(len(training_data0)):
        loss_evol0.append([torch.mean(training_data0[i][0]), torch.mean(training_data0[i][1])])
        loss_evol1.append([torch.mean(training_data1[i][0]), torch.mean(training_data1[i][1])])
    loss_evol0 = np.array(loss_evol0)
    loss_evol1 = np.array(loss_evol1)
    # plot these average losses
    fig1, (ax0, ax1)  = plt.subplots(1,2, figsize=(10,4)) 
    ax0.plot(loss_evol0[:, 0], '--', label='train loss', marker='x')
    ax0.plot(range(1, len(loss_evol0[:, 1])), loss_evol0[: -1, 1], '-.', label='test loss', marker='+')
    ax0.legend()
    ax0.set_title("Average losses for the simple AE")
    ax1.plot(loss_evol1[:, 0], '--', label='train loss', marker='x')
    ax1.plot(range(1, len(loss_evol1[:, 1])), loss_evol1[: -1, 1], '-.', label='test loss', marker='+')
    ax1.legend()
    ax1.set_title(" Average losses for the Deep AE")

    #--- plot the contour lines of the AE functions ---
    # construct the grid
    grid = np.linspace(-2,2,100)
    x_plot = np.outer(grid, np.ones(100))
    y_plot = np.outer(grid + 0.5, np.ones(100)).T
    potential_on_grid = np.zeros([100, 100])
    xi_ae0_on_grid = np.zeros([100, 100])
    xi_ae1_on_grid = np.zeros([100, 100])
    # compute values of potential and AEs on the grid
    for i in range(100):
        for j in range(100):
            x = np.array([grid[i], grid[j] + 0.5])
            potential_on_grid[i, j] = potential.V(x)
            xi_ae0_on_grid[i,j] = xi_ae(ae0, x)
            xi_ae1_on_grid[i,j] = xi_ae(ae1, x)
    # superimpose contour plots to colormap of the potential
    fig2, (ax0, ax1)  = plt.subplots(1,2, figsize=(9,3))        
    ax0.pcolormesh(x_plot, y_plot, potential_on_grid, cmap='coolwarm_r',shading='auto')
    ax0.contour(x_plot, y_plot, xi_ae0_on_grid, 20, cmap = 'viridis')
    ax0.set_title("Reaction coordinates with the simple AE")
    ax1.pcolormesh(x_plot, y_plot, potential_on_grid, cmap='coolwarm_r',shading='auto')
    ax1.contour(x_plot, y_plot, xi_ae1_on_grid, 20, cmap = 'viridis')
    ax1.set_title("Reaction coordinates with the Deep AE")
    return fig1, fig2

