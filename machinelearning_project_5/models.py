import nn

# 1. Neural Network
# A simple neural network has layers, where each layer performs a linear operation (just like perceptron). 
# Layers are separated by a non-linearity, which allows the network to approximate general functions. 


# We’ll use the ReLU operation for our non-linearity,
# . For example, a simple two-layer neural network for mapping an input row vector X to an output vector f(X)
#  would be given by the function: f(X) = reLu(X * W1 + b1) * W2 + b2
# W1 will be an i*h matrix, where i is the dimension of our input vectors X, and h is the hidden layer size. 
# b1 will be a size h vector

# Code:
# X*W = nn.Linear(X, W1)
# predicted_Y = nn.AddBias(X*W, b1)

# Using a larger hidden size will usually make the network more powerful (able to fit more training data), 
# but can make the network harder to train (since it adds more parameters to all the matrices and vectors we need to learn), 
# or can lead to overfitting on the training data. 

# We can also create deeper networks by adding more layers, for example a three-layer net

# 2. Batch
# For efficiency, you will be required to process whole batches of data at once rather than a single example at a time.
# This means that instead of a single input row vector x with size i, you will be presented with a batch of b inputs 
# represented as a b×i matrix 'X' as a batch of data samples

# 3. Tips
# Be systematic. 
# Keep a log of every architecture you’ve tried, what the hyperparameters (layer sizes, learning rate, etc.) were, 
# and what the resulting performance was. As you try more things, you can start seeing patterns about which parameters 
# matter. If you find a bug in your code, be sure to cross out past results that are invalid due to the bug.
#
# Start with a shallow network (just two layers, i.e. one non-linearity). Deeper networks have exponentially more 
# hyperparameter combinations, and getting even a single one wrong can ruin your performance. Use the small network 
# to find a good learning rate and layer size; afterwards you can consider adding more layers of similar size.
# 
# If your learning rate is wrong, 
# none of your other hyperparameter choices matter. A learning rate too low will 
# result in the model learning too slowly, and a learning rate too high may cause loss to diverge to infinity. 
# Begin by trying different learning rates while looking at how the loss decreases over time.
# 
# Smaller batches require lower learning rates. When experimenting with different batch sizes, be aware that the 
# best learning rate may be different depending on the batch size.
# 
# Refrain from making the network too wide (hidden layer sizes too large) If you keep making the network wider 
# accuracy will gradually decline, and computation time will increase quadratically in the layer size – you’re 
# likely to give up due to excessive slowness long before the accuracy falls too much. ---- not time efficient!!!
#
# If your model is returning Infinity or NaN, your learning rate is probably too high for your current architecture.

#******************************************
# IF TOO SLOW
# Learning rate too low/ hiddenlayer size too large/ Small batches
# IF INFINITY or NAN ( diverge to infinity.)
# Learning rate too high/ 

#Recommended values for your hyperparameters:
# Hidden layer sizes: between 10 and 400.
# Batch size: between 1 and the size of the dataset. For Q2 and Q3, we require that total size of the dataset be evenly divisible by the batch size.
# Learning rate: between 0.001 and 1.0.
# Number of hidden layers: between 1 and 3.

# Note:
# *******MULTIPLIER is learning rate in updating the weight, determine how much will weights learn from this training*******
# If you use a positive LR, the optimizer is performing gradient ascent (maximizing the loss) 
# rather than gradient descent (minimizing the loss) (negative learning rate!!!)


# nn.Constant is samples, (X is a Constant and Y is also Constant)
# nn.Parameter is parameters for nn, we use nn.Parameter(a, b) to construct a new unknown a*b size matrix Parameter
# The aim of training is to determine all the parameters unknown.
# batch_size * num_features: if num_features = 2, then dimension is 2, 2D classification
# batch_size is the num of samples in a batch, each row vector X is a sample
#
# Loss function is to evaluate if we should update the weights, cuz unlike binary Percetron, 
# the weight and X are not just vectors, they are matrix now so can not use DotProduct to determine if we should update weights
# So use nn.Linear(feature, weight) to get matrix multiplication.
# Then use it to evaluate the loss or update the weight or pass to next layer.
# 
# nn.gradients computes gradients of a loss with respect to provided parameters.
# Use nn.as_scalar can extract a Python floating-point number from a loss node. (Usage: nn.as_scalar(node), where node is either a loss node or has shape (1,1).)

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        accuracy = 0.0
        batch_size = 1
        while accuracy != 1:
            num_correct = 0
            num_samples = 0
            for x, y in dataset.iterate_once(batch_size):
                num_samples += 1
                predx = self.get_prediction(x)
                predy = nn.as_scalar(y)              
                if predx == predy:
                    num_correct += 1
                else:
                    nn.Parameter.update(self=self.get_weights(), direction=x, multiplier=predy)
            accuracy = num_correct / num_samples
            # To receive full points for this question, your perceptron must converge to 100% accuracy

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Origin Input: batch_size × original_num_feature: depends on data, 
        # in this case, it's sin(x), so num_feature(dimension) is 1
        # Hidden layer sizes: between 10 and 400. : Used as Inbetween features Numbers
        # weights of shape num_input_features × num_output_features of current layer
        # bias of shape 1 × num_features
        # layer input features of shape batch_size × num_input_features
        # layer out of shape batch_size × num_out_features
        # two layers
        self.batch_size = 200
        self.w1 = nn.Parameter(1, 100) # inital feature is only 1, cuz it's sin(x), and hidden layer size is 100
        self.w2 = nn.Parameter(100, 1)
        self.b1 = nn.Parameter(1, 100)
        self.b2 = nn.Parameter(1, 1)
        self.rate = -0.005 # learning rate



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1)) # output batchsize * numoutfeatures, 100 * 100
        layer2 = nn.AddBias(nn.Linear(layer1, self.w2), self.b2)
        return layer2


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        lossScore = float('inf')
        # You may use the training loss to determine when to stop training 
        # (use nn.as_scalar to convert a loss node to a Python number)

        # these two are the same, later is better of lower loss score
        # for x, y in dataset.iterate_forever(self.batch_size):
        #     loss = self.get_loss(x, y)
        #     lossScore = nn.as_scalar(loss)
        #     if (lossScore > 0.02):
        #         gradient = nn.gradients(loss, [self.w1, self.w2, self.b1, self.b2])
        #         self.w1.update(gradient[0], self.rate)
        #         self.w2.update(gradient[1], self.rate)
        #         self.b1.update(gradient[2], self.rate)
        #         self.b2.update(gradient[3], self.rate)
        #     else:
        #         return
        while lossScore > 0.02:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                lossScore = nn.as_scalar(loss)
                if (lossScore > 0.02):
                    gradient = nn.gradients(loss, [self.w1, self.w2, self.b1, self.b2])
                    self.w1.update(gradient[0], self.rate)
                    self.w2.update(gradient[1], self.rate)
                    self.b1.update(gradient[2], self.rate)
                    self.b2.update(gradient[3], self.rate)
        # If your model is returning Infinity or NaN, your learning rate is probably too high for your current architecture.
        # negative rate works well too


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.expectAccurancy = 0.975
        self.batch_size = 100
        # 3 layers
        self.w1 = nn.Parameter(784, 100) # hidden layer should not be too wide
        self.w2 = nn.Parameter(100, 40)
        self.w3 = nn.Parameter(40, 10) #output is 10 num features/vector
        self.b1 = nn.Parameter(1, 100)
        self.b2 = nn.Parameter(1, 40)
        self.b3 = nn.Parameter(1, 10)
        self.rate = -0.5 
        # If you use a positive LR, the optimizer is performing gradient ascent (maximizing the loss) 
        # rather than gradient descent (minimizing the loss) (negative learning rate!!!)



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # Do not put a ReLU activation after the last layer of the network.
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        layer2 = nn.ReLU(nn.AddBias(nn.Linear(layer1, self.w2), self.b2))
        layer3 = nn.AddBias(nn.Linear(layer2, self.w3), self.b3)
        return layer3 #we don't have to determine which one is true, alg will predict for us based on (x, y (true label))

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # You can use dataset.get_validation_accuracy() to compute validation accuracy for your model, 
        # which can be useful when deciding whether to stop training.
        accuracy_score = dataset.get_validation_accuracy()
        while accuracy_score < self.expectAccurancy:
            for x, y in dataset.iterate_once(self.batch_size):
                preD = self.run(x)
                if preD != y:
                    loss = self.get_loss(x, y)
                    gradient = nn.gradients(loss, [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])
                    self.w1.update(gradient[0], self.rate)
                    self.w2.update(gradient[1], self.rate)
                    self.w3.update(gradient[2], self.rate)
                    self.b1.update(gradient[3], self.rate)
                    self.b2.update(gradient[4], self.rate)
                    self.b3.update(gradient[5], self.rate)
                accuracy_score = dataset.get_validation_accuracy()



# Recurrent Neural Network (RNN) to encode complicated input (the arbitrary-length input word) 
# into a fixed-size vector and put it to additional layers for classification
# (fed through additional output layers to generate classification scores for the word’s language identity.)

# RNN is just NN with multi inputs affecting each other

# our code in the project ensures that all words within a single batch have the same length. 
# so that we could do batch_size * hiddenstatesSize hi
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 100
        self.expectAccurancy = 0.83
        # Start with a shallow network for f, and figure out good values for the hidden size and 
        # learning rate before you make the network deeper.

        # Input List is XS, input element is batch_size * num_chars (features)
        # hidden_size is fixed and should be sufficient large
        self.hidden_size = 200 
        # should be too much, too large hidden layer will affect accuracy!!!!!!
        self.wInitial = nn.Parameter(self.num_chars, self.hidden_size)
        self.wHidden = nn.Parameter(self.hidden_size, self.hidden_size) 
        self.w = nn.Parameter(self.hidden_size, 5) # output size : 5 languages
        # output of hidden_layer H should be batch_size * hidden_size, and 
        # z = nn.Add(nn.Linear(x, W), nn.Linear(h, W_hidden)) should also be the output of each hidden layer, 
        # so Weight of hidden should be hidden_size * hidden_size
        self.bInitial = nn.Parameter(1, self.hidden_size)
        self.b = nn.Parameter(1, 5)
        self.rate = -0.05 # too small --> too slow


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] (the second letter of 'CAT') will be a node that contains a 1 at position (7, 0). 
        Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th on Alphabet)
        letter of our combined alphabet for this task.
        
        each element is batchsize * num_alphabet
        First element is about for all first letters of all words in this batch, How they distributed on alphabet!!
        if there is '1', at one row (one word in this batch), at one col (this letter on alphabet), 
        then it's the letter of first letter of this word. All other cols on this row should be '0'
        a b c d e f g h d d d d d

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars) (input)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # You can implement this using a for loop over the provided inputs xs
        # resulting function f(x,h)=g(zx,h) will be non-linear in both x and h
        fInitial = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.wInitial), self.bInitial))
        fCurrent = fInitial
        for letter in xs:
            if letter != xs[0]:
                fCurrent = nn.ReLU(nn.Add(nn.AddBias(nn.Linear(letter, self.wInitial), self.bInitial), nn.Linear(fCurrent, self.wHidden)))
        fFinal = nn.AddBias(nn.Linear(fCurrent, self.w), self.b)
        # don't use ReLU for the last layer
        return fFinal

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accuracy = dataset.get_validation_accuracy()
        while accuracy < self.expectAccurancy:
            for xs, y in dataset.iterate_once(self.batch_size):
                preD = self.run(xs)
                if preD != y:
                    loss = self.get_loss(xs, y)
                    gradient = nn.gradients(loss, [self.w, self.wInitial, self.wHidden, self.b, self.bInitial])
                    self.w.update(gradient[0], self.rate)
                    self.wInitial.update(gradient[1], self.rate)
                    self.wHidden.update(gradient[2], self.rate)
                    self.b.update(gradient[3], self.rate)
                    self.bInitial.update(gradient[4], self.rate)
                accuracy = dataset.get_validation_accuracy()

