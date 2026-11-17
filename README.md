# nn-playground
Implementation of a NN using only numpy to play around and understand new stuff. Training the networks requires downloading the MNIST dataset in the format presented in Michael Nielsen's _Neural networks and deep learning_. 

- nn_layer.ipynb contains a single layer of a NN and preliminary experiments with it
- nn.ipynb contains an implementation of a full network connecting instances of the single nn_layer and experiments with it. So far, i've included my own version of backprop and adaptations to multiple loss functions
- utils.py contains helper functions like activation functions and plotting functions
- dec_to_bin.py is just a network with hand-picked weights to convert numbers from decimal to binary representation
