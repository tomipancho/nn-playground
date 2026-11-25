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

def plot_loss(train_losses, val_losses, title='Loss over Epochs'):
    epochs = range(len(train_losses))
    plt.figure()
    plt.plot(epochs, train_losses, label='training loss')
    plt.plot(epochs, val_losses, label='validation loss')
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()

def plot_accuracies(train_accuracies, val_accuracies, title='Accuracy over Epochs'):
    epochs = range(len(train_accuracies))
    plt.figure()
    plt.plot(epochs, train_accuracies, label='training accuracy')
    plt.plot(epochs, val_accuracies, label='validation accuracy')
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
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
            g = layers[-1].del_activation(args[-1]) * res # (n_out, d)


        if loss=="CE":
            # softmax output layer and CE loss (log-likelihood loss in Nielsen) 
            from scipy.special import softmax
            p = softmax(args[-1], axis=0) # (n_out, d)
            res = y - p # (n_out, d) 

            # Compute first error vector
            g = layers[-1].del_activation(args[-1]) * res # (n_out, d)


        db = - np.reshape(np.mean(g, axis=1), (args[-1].shape[0], 1)) # (n_out, 1) 
        dW = - g @ zs[-2].T / d # (n_curr, n_prev)

        del_b.append(db)
        del_W.append(dW)


        # Iterate over layers backwards
        for i in range(L - 1):

            # To count backwards, we use j from L-2 to 0 (L - 1-th layer to 1-st layer)
            j = L - 2 - i

            # Edge case, input layer
            if j == 0:
                z_prev = x
            else:
                z_prev = zs[j - 1]
            
            # Extract output of previous layer and current layer's arg
            arg = args[j] # (n_curr, d)
            n_curr = arg.shape[0]

            # At the output layer, construct the first g
     
            g = ((layers[j + 1].W).T @ g) * layers[j].del_activation(args[j])
            # Use g to get partial derivatives
            db = - np.reshape(np.mean(g, axis=1), (n_curr, 1)) # (n_out, 1) 
            dW = - g @ z_prev.T / d # (n_curr, n_)

            del_b.append(db)
            del_W.append(dW)
            
        
        del_W = del_W[::-1]
        del_b = del_b[::-1]

        return del_W, del_b

         
        


def compute_accuracy(NN, x, true_labels):

    pred_onehot = NN(x)[-1]
    pred_labels = onehot_to_labels(pred_onehot)

    accuracy = np.mean(true_labels == pred_labels)
    return accuracy


def train(NN, train_data, val_data, epochs=20, minibatch_size=50, eta=0.1, verbose=True, loss="CE", lam=0):
    '''
    Training function for a single layer using SGD. Must include a test/train split. Performs accuracies and loss calculations 
    after each epoch. Returns a NN with updated parameters and arrays containing train and test cost and accuracies.

    - L is a layer as defined above. Initial weights and biases already initialized.
    - training_data is 
    - epochs is the number of times that the training loop will visit the full training set 
    - minibatch_size is the size of the minibatch. *Same batching every epoch* TO DO: randomize batches
    - eta is the learning rate
    - verbose is a boolean indicating the desire of outputing intermediate loss calculations after each epoch
    - loss is a string indicating the loss that is used for training
    - lam is the L2 regularization parameter

    '''
    # Compute initial training loss
    loss0 = NN.CE_loss(train_data)
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    train_accuracies = np.zeros(epochs)
    val_accuracies = np.zeros(epochs)
    print('Initial Training Loss', loss0)

    x_train = train_data[0]
    y_train = train_data[1]

    # Compute number of minibatches
    d_train = x_train.shape[1]
    n_batches = int(d_train / minibatch_size)
    
    for e in tqdm(range(epochs)):

        # Shuffle training data
        idx = np.random.permutation(d_train)

        x_train = x_train[:, idx]
        y_train = y_train[:, idx]

        for n in range(n_batches):

            # Organize batch's training samples for SGD
            # Edge case: last batch may not be big enough. Avoid slicing error
            if n == n_batches - 1:
                x_batch = x_train[:, n * minibatch_size :]
                y_batch = y_train[:, n * minibatch_size :]
            else:
                x_batch = x_train[:, n * minibatch_size : (n+1) * minibatch_size]
                y_batch = y_train[:, n * minibatch_size: (n+1) * minibatch_size]

            data_batch = (x_batch, y_batch)

            # Compute gradients   
            del_W, del_b = NN.backprop(data_batch, loss=loss)

            # Iterate over layers and update parameters
            for i in range(len(NN.layers)):
                
                # Extract layer gradients
                del_Wi, del_bi = del_W[i], del_b[i]
                
                # Gradient step with (optional) L2 regularization
                NN.layers[i].W = (1 - lam * eta / d_train) * NN.layers[i].W - eta * del_Wi
                NN.layers[i].b = NN.layers[i].b - eta * del_bi


        # Recompute loss at the end of the epoch

        train_loss = NN.CE_loss(train_data)
        val_loss = NN.CE_loss(val_data)
        train_losses[e] = train_loss
        val_losses[e] = val_loss

        # Compute training and validation accuracies at each epoch
        train_acc = compute_accuracy(NN, x_train, onehot_to_labels(y_train))
        val_acc = compute_accuracy(NN, val_data[0], onehot_to_labels(val_data[1]))
        train_accuracies[e] = train_acc
        val_accuracies[e] = val_acc

        if verbose==True:
            print('Loss at epoch e =', e, ':', train_loss)

    return NN, train_losses, val_losses, train_accuracies, val_accuracies

