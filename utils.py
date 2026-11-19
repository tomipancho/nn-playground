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

def id(z):
    ''' 
    Identity activation function
    '''
    return z

def del_id(z):
    '''
    Derivative of identity activation
    '''
    return np.ones_like(z)

def ReLU(z):
    ''' 
    ReLU activation function
    '''
    return np.maximum(0, z)

def del_ReLU(z):
    ''' 
    Derivative of ReLU activation
    '''
    return (z > 0).astype(np.float64)

class layer:

    def __init__(self, n_in, n_out, activation=sigmoid, del_activation=del_sigmoid) -> None:
        ''' 
        Initialize layer with a random weight matrix and a random bias
        '''

        #self.W = np.random.uniform(-1, 1, (n_out, n_in)) # (n_out, n_in)
        self.W = np.random.randn(n_out, n_in) *  np.sqrt(2 / n_in)
        self.activation = activation 
        self.del_activation = del_activation
        #self.b = np.random.uniform(-1, 1, (n_out, 1)) # (n_out, 1)
        self.b = np.zeros((n_out, 1))
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
        return self.activation(arg), arg # (n_out)

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
        phi_derivative = self.del_activation(arg)
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
        phi_derivative = self.del_activation(arg) # (n_out, d)

        # Reshape arrays for multiplication
        W = np.reshape(W, (n_out, n_in, 1))
        phi_derivative = np.reshape(phi_derivative, (n_out, 1, d))

        # Compute Jacobian
        J = phi_derivative * W # (n_out, n_in, d)

        # Assertion for correct shape
        assert J.shape == (n_out, n_in, d) 

        return J

class network:

    def __init__(self, spec, activation=[sigmoid], del_activation=[del_sigmoid]):
        ''' 
        Initialization of a neural network. spec is a list of numbers specifying the size of each layer. 
        Start from the input layer and finish with the output layer.

        An activation is a layer's quality. Should not be a network's attribute.

        MODIFICATION: Allow for layers with different activations.
        '''
        self.layers = []

        assert len(activation) == len(del_activation), "Must provide same number of activations and derivatives"

        if len(activation) > 1:
            assert len(spec) == len(activation) + 2, 'Input and output layer are not activated'

        for i in range(len(spec) - 1):

            # Extract input and output sizes of the i-th layer
            n_in = spec[i]
            n_out = spec[i + 1]

            # Output layer: no activation
            if i == len(spec) - 2:
                act = id  
                del_act = del_id

            # Hidden layers 
            elif len(activation) == 1:
                act = activation[0]
                del_act = del_activation[0]

            else:
                act = activation[i]
                del_act = del_activation[i]

            self.layers.append(
                layer(n_in, n_out, 
                        activation=act,
                        del_activation=del_act)
                        )

    def __call__(self, x, return_args=False): 
        '''
        Network's forward pass. Is there any way around sequentiallity? Or do I necessarilly need to use for loops
        x has a shape (n_in, d). We wish to pass multiple inputs through the network at once. We save intermediate outputs
        '''
        
        # Assert input's correct size
        layers = self.layers
        L0 = layers[0]
        n_in = L0.dims[1]
        d = x.shape[1]
        assert x.shape[0] == n_in, 'input size is incorrect'

        # Initialize array to save intermediate outputs
        zs = []
        args = []

        # Loop over each layer's output
        for i in range(len(layers)):

            # Edge case: first layer takes a data-point as input
            if i == 0:
                z, arg = layers[i](x) # (n_1, d)
                assert z.shape[0] == layers[i].dims[0], 'output of first layer has incorrect size'
                zs.append(z)
                args.append(arg)
                continue

            # For the rest of the network, the input of a layer is the output of the previous layer
            z, arg = self.layers[i](z) # (n_i, d)

            zs.append(z)
            args.append(arg)

        # Return the full intermediate outputs array. The final output is just zs[-1]
        if return_args==True:
            return zs, args
        
        return zs

    def MSE_loss(self, data):
        ''' 
        Loss function of the network with respect to a set of data.  
        Data is of the form (x, y), where x is an array of shape (n_in, d) and
        y has the shape (n_out, d)
        '''

        # Get Network's forward pass
        x = data[0] # (n_in, d)
        y = data[1] # (n_out, d)
        d = x.shape[1]
        layers = self.layers

        # Assert the output has correct size
        Lout = layers[-1]
        n_out = Lout.dims[0]
        assert n_out == y.shape[0], 'size of last layers output and the size of data targets dont match'
        assert d == y.shape[1], 'number of inputs and targets in the dataset dont match'

        # Network's forward pass
        zs = self(x)
        C = zs[-1] # (n_out, d)

        # Compute residuals 
        res = y - C # (n_out, d)
        mags = np.linalg.norm(res, axis=0) ** 2 # (d,)

        # Compute loss
        loss = 0.5 * np.mean(mags) # Just a number 
        return loss


    def grad_loss(self, data):
        ''' 
        Computes the partial derivatives of the loss function with respect to all parameters in the network. 
        Returns an two arrays filled with the partial derivatives with respect to each layer's weight matrix and bias vector
        shape (del_W, del_b) where del_W is a list of matrices and del_b is a list of vectors.
        '''
        # Get data
        x = data[0]
        y = data[1]
        d = x.shape[1]
        layers = self.layers
        

        # compute network's forward and residuals
        zs = self(x)
        res = y - zs[-1] # (n_out, d)

        # initialize lists to store gradients
        del_W = []
        del_b = []

        # if its a single layer, use gradients from before
        if len(layers) == 1:
            dW, db = layers[0].grad_loss(data)
            del_W.append(dW)
            del_b.append(db)
            return del_W, del_b  

        # Iterate over layers, start with the last layer
        for i in range(len(layers)):

            # Invert counting order
            j = len(layers) - 1 - i

            # layer output and params
            z = zs[j] # (n_curr, d)
            (n_curr, n_prev) = layers[j].dims 
            assert z.shape[0] == n_curr, 'wrong layer output dimensions'

            # Edge case: don't compute jacobian for the output layer
            if j == len(layers) - 1:

                # Output layer params
                n_out = n_curr
                n_sec_to_last = n_prev
                
                # Use previous layer code to compute gradient.  
                data_out = (zs[j - 1], y) 
                del_W_out, del_b_out = layers[j].grad_loss(data_out) 
                del_W.append(del_W_out) # (n_out, n_prev)
                del_b.append(del_b_out) # (n_out,)

                # Initial jacobian
                H = layers[j].jacobian(zs[j - 1]) # (n_out, n_prev, d)

                continue

            # assert shape of current jacobian
            assert H.shape == (n_out, n_curr, d), 'wrong jacobian shape'
 
            # activation derivative and forward derivative 
            del_phi = np.reshape(z * (1 - z), (1, n_curr, d)) # (1, n_curr, d)
            mult = H * del_phi # (n_out, n_curr, d) 

            # build partial derivatives for bias 
            inner_b = np.einsum('ij, ikj -> kj', res, mult) # (n_curr, d)
            del_b_curr = - np.mean(inner_b, axis=1) # (n_curr,)
            del_b_curr = np.reshape(del_b_curr, (n_curr, 1))

            # build partial derivatives for weights, edge case for first layer 
            if j == 0:
                z_prev = x # (n_in, d)
            else:
                z_prev = zs[j - 1] # (n_prev, d)

            # multiply forward derivative with previous output
            del_W_curr = -(inner_b @ z_prev.T) / d   # (n_curr, n_prev)

            # Store partial derivatives
            del_W.append(del_W_curr)
            del_b.append(del_b_curr)
    
            if j > 0:
                J = layers[j].jacobian(z_prev)                 # (n_curr, n_prev, d)
                H = np.einsum('ijk, jnk -> ink', H, J)         # (n_out, n_prev, d)

        # Reverse order of lists
        del_W = del_W[::-1]
        del_b = del_b[::-1]

        return del_W, del_b

    def CE_loss(self, data):
        ''' 
        Computes the CE loss of the dataset. From the output of the network, we can use the resulting logits under a softmax to get
        a probability distribution. We use this distribution to compute the average CE loss, that is, the negative logprob of having correctly classified
        each label. 

        data = (x, y) where x is (n_in, d) and y has shape (n_out, d) one-hot vectors
        '''
        x = data[0]
        y = data[1]
        d = x.shape[1]

        # Forward pass of the NN - logits
        zs, args = self(x, return_args=True)
        logits = args[-1] # (n_out, d)

        # Compute CE loss using logsumexp form 
        from scipy.special import logsumexp

        y_labels = onehot_to_labels(y).astype(int) # (d,)
        sample_loss = -logits[y_labels, np.arange(d)] + logsumexp(logits, axis=0) # (d,)

        return np.mean(sample_loss) 


    def backprop(self, data, loss="MSE"):
        ''' 
        Computation of the loss' gradient using a (hopefully) faster algorithm. Looks promising. No need for computing huge jacobians.
        We only pass one vector from layer to layer and update via matrix multiplication with the weight matrix. 

        data = (x, y) where x has shape (n_in, d) and y has shape (n_out, d)
        '''
        x = data[0]
        y = data[1]
        d = x.shape[1]
        layers = self.layers

        # Prepare lists to store each layer's partial derivatives
        del_W = []
        del_b = []

        # Compute network's output and residual
        zs, args = self(x, return_args=True) 
        L = len(zs)

        if loss=="MSE":
            res = y - zs[-1] # (n_out, d)

        if loss=="CE":
            # compute softmax of output 
            from scipy.special import softmax
            p = softmax(args[-1], axis=0) # (n_out, d)
            res = y - p # (n_out, d) 

        # Iterate over layers backwards
        for i in range(L):

            # To count backwards, we use j from L-1 to 0 (L-th layer to 1-st layer)
            j = L - 1 - i

            # Edge case, input layer
            if j == 0:
                z_prev = x
            else:
                z_prev = zs[j - 1]
            
            # Extract output of previous layer and current layer's arg
            n_prev = z_prev.shape[0]
            arg = args[j] # (n_curr, d)
            n_curr = arg.shape[0]

            # At the output layer, construct the first g
            if j == L - 1:

                # Get first g and compute partial derivatives at the output layer
                g = layers[j].del_activation(arg) * res # (n_out, d)
                gr = np.reshape(g, (n_curr, 1, d)) # (n_out, 1, d)
                
                db = - np.mean(gr, axis=2) # (n_out, 1) 
                dW = - g @ z_prev.T / d

                del_b.append(db)
                del_W.append(dW)

                h = (layers[j].W).T @ g
                
                continue

            g = layers[j].del_activation(arg) * h # (n_curr, d)
            gr = np.reshape(g, (n_curr,1,d)) # (n_curr, 1, d)

            # Use g to get partial derivatives
            db = - np.mean(gr, axis=2) # (n_out, 1) 
            dW = - g @ z_prev.T / d # (n_curr, n_)

            del_b.append(db)
            del_W.append(dW)
            
            if j > 0:
                # if we are at the input layer, there is no need to recompute. First weight matrix never appears. Interesting. 
                h = (layers[j].W).T @ g
        
        del_W = del_W[::-1]
        del_b = del_b[::-1]

        return del_W, del_b





def compute_accuracy(NN, x, true_labels):

    pred_onehot = NN(x)[-1]
    pred_labels = onehot_to_labels(pred_onehot)

    accuracy = np.mean(true_labels == pred_labels)
    return accuracy


