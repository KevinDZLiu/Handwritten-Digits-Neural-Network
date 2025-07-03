import random
import numpy as np


class Network(object):

    def __init__(self, neurons):
        """
        neurons: list containing number of neurons in the
        respective layers of the network. e.g. [2, 3, 1] is a three
        layer network, with 2 neurons, 3 neurons, 1 neuron.
        First layer input layer.
        biases and weights initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.
        """
        self.num_layers = len(neurons)
        self.neurons = neurons
        # randn(d0, d1, ...) is dimension of sample e.g. randn(2, 1) is 2x1
        # matrix with each entry a random sample
        # bias is for each neuron, apart from inputs
        self.biases = [np.random.randn(n, 1) for n in neurons[1:]]
        # weights connect previous layer neurons to next layer neurons
        # (y, x) because e.g. if from x = 2 neuron to y = 3 neurons, then
        # matrix is 3x2, then when multiplied by activation of each neuron in
        # layer x, which is column vector, then will result in column vector
        # of sum of weights times activation for each neuron in layer y
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(neurons[:-1], neurons[1:])]
        
    def feedforward(self, a):
        """
        Returns next layers a' for input a of previous layer
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Stochastic Gradient Descent
        Training neural network using mini-batch stochastic
        gradient descent. 

        epochs: how many times to go through all the training data
        eta: learning rate

        If test_data provided then network will be evaluated against
        the test data after each epoch, and partial progress printed out.  
        This is useful for tracking progress, but slows things down 
        substantially.
        """
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] 
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {i}: {self.evaluate(test_data)} / {n_test}")
            else:
                print("Epoch {0} complete".format(i))

    def update_mini_batch(self, mini_batch, eta):
        """
        Update weights and biases by applying gradient descent
        using backpropagation to a single mini batch.
        mini_batch: list of tuples (x, y)
        eta: learning rate.
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_grad_b, delta_grad_w = self.backprop(x, y)
            grad_b = [nb+dnb for nb, dnb in zip(grad_b, delta_grad_b)]
            grad_w = [nw+dnw for nw, dnw in zip(grad_w, delta_grad_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, grad_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, grad_b)]

    def backprop(self, x, y):
        """
        Return a tuple (grad_b, grad_w) representing the
        gradient for the cost function C_x.
        grad_b and grad_w are layer-by-layer lists of numpy arrays, similar
        to self.biases and self.weights.
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
            grad_b[-layer] = delta
            grad_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
        return (grad_b, grad_w)
    
    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives partial C_x /
        partial a for the output activations.
        """
        return (output_activations-y)

    def evaluate(self, test_data):
        """
        Return number of test inputs for which the neural
        network outputs the correct result.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
