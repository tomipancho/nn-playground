import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 

# In this file I will include all classes and functions that I know are working correctly. 
# Notebooks are for exploration, but to use the functions that in other files/projects this .py script is more appropriate. 

# One-Hot encoding minutia

def labels_to_onehot(labels):
    '''
    Produces one-hot encodings of a set of labels
    '''
    d_size = len(labels)
    one_hot = np.zeros((10, d_size))

    # Fill up one-hot encodings
    for n in range(d_size):
        one_hot[labels[n], n] = 1

    return one_hot

def onehot_to_labels(onehot):
    '''
    Produces labels from one-hot encodings
    '''
    d_size = onehot.shape[1]
    labels = np.zeros(d_size)

    # Fill up labels
    for n in range(d_size):
        labels[n] = np.argmax(onehot[:,n])

    return labels

# Plotting

def plot_loss(train_losses, val_losses):
    epochs = range(len(train_losses))
    plt.figure()
    plt.plot(epochs, train_losses, label='training loss')
    plt.plot(epochs, val_losses, label='validation loss')
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


# Data Preprocessing

def test_train_split(data, split):
    ''' 
    Divides the data in test and training sets according to the split. 
    split is a number between 0 and 100 defining the percentage of training data.
    '''
    x = data[0]
    y = data[1]
    d = x.shape[1]
    idxs = np.arange(d)

    # number of training samples
    n_train = round(d * split / 100) 

    # sample indices for training set 
    np.random.seed(seed=42)
    idxs_train = np.random.randint(0, d, n_train)

    # remaining indices belong to the test set   
    idxs_test = np.array([])
    for i in range(d):
        if i not in idxs_train:
            np.append(idxs_test, i)

    # Training Set
    x_train = x[idxs_train]
    y_train = y[idxs_train]

    # Test Set
    x_test = x[idxs_test]
    y_test = y[idxs_test]

    # Consistency test
    assert x_test.shape[0] + x_train.shape[0] == x.shape[0]
    assert y_test.shape[0] + y_train.shape[0] == y.shape[0]

    return (x_train, y_train), (x_test, y_test)



# Neural Network Layer

def sigmoid(z):
    '''
    Activation Function
    '''
    return 1/(1 + np.exp(-z))

def del_sigmoid(z):
    ''' 
    Derivative of the activation function 
    '''
    phi = sigmoid(z)
    return phi * (1 - phi) 


class layer:

    def __init__(self, n_in, n_out, activation=sigmoid, del_activation=del_sigmoid) -> None:
        ''' 
        Initialize layer with a random weight matrix and a random bias
        '''

        self.W = np.random.uniform(-1, 1, (n_out, n_in)) # (n_out, n_in)
        self.phi = activation 
        self.del_phi = del_activation
        self.b = np.random.uniform(-1, 1, (n_out, 1)) # (n_out, 1)
        self.dims = (n_out, n_in)

        pass

    def __call__(self, x):
        ''' 
        Forward pass of the neural network layer. 
        NN(x) = phi(Wx + b)
        x is an array of the shape (n_in, d)
        '''
        Wx = self.W @ x
        arg = Wx + self.b
        return self.phi(arg), arg # (n_out)

    def loss(self, data):
        ''' 
        Loss function. With respect to any set of data
        '''
        # retrieve datapoints
        x = data[0] # (n_in, d)
        y = data[1] # (n_out, d)
        d = x.shape[1]
        dp = y.shape[1]
        assert d == dp, 'data shapes of x and y dont match'

        # In and output vector dimensions
        n_in = x.shape[0]
        n_out = y.shape[0]

        # Forward pass and residuals (n_out, d)
        C, arg = self(x)
        res = y - C 

        # Compute magnitude squared of each residual
        mags = np.linalg.norm(res, axis=0) ** 2 # (d,) 
        assert mags.shape[0] == d

        # Return mean magnitude squared 
        return 0.5 * np.mean(mags)


    def grad_loss(self, data):
        '''
        Gradient of the loss function as I've computed it. 
        Requires data as a tuple (x, y) where x is an array of the form (n_in, d),
        and y is an array of shape (n_out, d). 
        d is the number of data points in the dataset.

        Returns a vector del_b of shape (n_out,) and a matrix del_W of shape (n_out, n_in)  
        '''
        x = data[0] # (n_in, d)
        y = data[1] # (n_out, d)
        d = x.shape[1]
        dp = y.shape[1]
        assert d == dp, 'data shapes of x and y dont match'
        n_in = x.shape[0]
        n_out = y.shape[0]

        # Forward pass, residuals and component-wise multiplication (n_out, d)
        C, arg = self(x)
        res = y - C 
        phi_derivative = self.del_phi(arg)
        mult = res * phi_derivative  

        # Summation over mult and averaging. I guess this is the most computationally-intensive part.  
        del_b = - np.reshape(np.mean(mult, axis=1), (-1,1)) # (n_out, 1)
        del_W = -1 / d * np.einsum('ij,kj -> ik', mult, x) # (n_out, n_in) 

        return del_W, del_b
    
    def jacobian(self, x):
        ''' 
        Jacobian of the layer. This is the total derivative of the layer with respect to its input x. Not its parameters
        From our calculations, J(x) = diag(phi'(Wx + b)) W 

        If we want to compute the full jacobian with respect to all inputs in the dataset, we can try to return an array of shape
        (n_out, n_in, d)
        '''

        # Compute layer's argument and output 
        C, arg = self(x) # (n_out, d). Make sure x always has this shape
        W = self.W.copy()
        n_in = x.shape[0]
        n_out = C.shape[0]
        d = x.shape[1]

        # Compute Activation Derivatives 
        phi_derivative = self.del_phi(arg) # (n_out, d)

        # Reshape arrays for multiplication
        W = np.reshape(W, (n_out, n_in, 1))
        phi_derivative = np.reshape(phi_derivative, (n_out, 1, d))

        # Compute Jacobian
        J = phi_derivative * W # (n_out, n_in, d)

        # Assertion for correct shape
        assert J.shape == (n_out, n_in, d) 

        return J


def compute_accuracy(NN, x, true_labels):

    pred_onehot = NN(x)[-1]
    pred_labels = onehot_to_labels(pred_onehot)

    accuracy = np.mean(true_labels == pred_labels)
    return accuracy


