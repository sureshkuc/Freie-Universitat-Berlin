Project1: Implementation of a k-NN-Classifier
Implement a k-NN-Classifier in Python (incl. Numpy, Pandas, Matplotlib) on the Jupyter Notebook Environment. Use the “ZIP-Code”-Dataset1 with the training data as reference
for neighborhood. Evaluate the model on the test data.
(a) Print out the accuracy.
(b) Using Matplotlib, plot some of the numbers that are classified incorrectly.
(c) Which k is optimal for the accuracy?
(d) What are advantages and disadvantages for the k-NN-Classifier?

Project2: Implementation of a DBSCAN-Classifier
Implement a DBSCAN-Clustering in Python (incl. Numpy, Matplotlib) on the Jupyter Notebook Environment. Apply the algorithm on the “Two-Spirals”-dataset given in the notebook.
Evaluate the model on points given by that distribution.
(a) Use Mathplotlib to create a scatter plot highlighting the clusters that were found after
finding good hyperparameter values eps and minPts.
(b) Print accuracies for different data_size values.
(c) For what kind of data_size values does the algorithm fail and why? What would you
say are disadvantages of DBSCAN?

Project3: Logistic Regression
Logistic Regression can be interpreted as a neural network with just one layer. It uses the
Cross Entropy to measure the performance of the layer (i.e. of the ”trained” weight w and
bias b). In ML we call this the Loss function.
Implement Logistic Regression using Python (incl. Numpy etc.) and use it on the ” ZIPCode”-Dataset 2
. Implement the Cross Entropy and the Sigmoid function from scratch. Use
gradient descent to optimize.
(a) What happens when you take the Means Squared Error (MSE) instead of the Cross
Entropy? Does this also work? Implement MSE and try for yourself.
(b) (Optional) Can you think of a way to classify more than one class (in this case 10
classes)? How would you change the way w and b is defined?

Project4: Implement soft-margin Support Vector Machine
algorithm, need to train and test on Iris Dataset and clean the Dataset provided.

Project5: Need to implement, estimate and compare the variance
on different sample sizes using different sampling techniques on the given dataset
ie. Simple random sampling without replacement and stratified sampling.

Project6: 
Exercise 1. Decision Trees
Implement a decision tree (classification tree to be precise) using Python (incl. Numpy etc.)
and use it on the SPAM-Dataset1
. Use a metric of your choice as a loss function.
(a) Assume that classifying a genuine E-Mail as spam is ten times worse than classifying
spam as genuine. How would you change the design of your decision tree?
(b) Use your tree to analyze feature importance. Plot the difference between the top 5
features (check spambase.names to check what features those belong to).
Exercise 2. Random Forests
Implement a Random Forest and use it on the SPAM-Dataset.
(a) Print a confusion matrix (you can use package implementations here).
(b) What is a good number of trees in the forest?

Project7:
Excercise 1. AdaBoost
Implement AdaBoost using Python (incl. Numpy etc.) and use it on the SPAM-Dataset1
The weak classifiers should be decision stumps (i.e. decision trees with one node).
(a) Print a confusion matrix.
(b) Is AdaBoost better when using stronger weak learners? Why or why not? Compare
your results to using depth-2 decision trees.
Excercise 2 (Bonus). Viola-Jones Face Detection
Implement the Viola-Jones algorithm (without the cascade mechanism) and use it on a
LFW-Face-subset2
to classify faces.
(a) Visualize the top ten face classifiers.
Excercise 3 (Bonus). Cascade-Classification
Implement a cascade algorithm to classify faces in a picture of your choice (there should be
more than a face on your image, e.g. skimage.data.astronaut())

Project8: 
Excercise 1. Perceptron
Implement the Perceptron algorithm using Python (incl. Numpy etc.) and use it on the
Iris-Dataset1
. Train the algorithm to seperate Setosa from Versicolour and Virginica.
(a) What happens if you use the algorithm to seperate Versicolour from Virginica? (Evaluate multiple runs)
(b) Find a way to solve the problem and obtain the accuracy.
Excercise 2. Multilayer-Perceptron (MLP)
Implement a class that builds an MLP with both variable depth D (number of layers) and
variable number of neurons ni
for each layer i = 1, ..., D. Produce outputs on the ZIPDataset2

Project9:
Excercise 1. Backpropagation
Add Backpropagation to your MLP and train the model on the ZIP-Dataset1
.
(a) Optimize width (the number of neurons in a hidden layer; it is usually the same for all
of them) and depth of the network. Try to find a setting that trains in a reasonable
time. Plot the loss.
(b) Show some digits that are classified incorrectly.
(c) Plot your first weight layer as a grayscale image.

Project10:
Excercise 1. Convolutional Neural Networks (CNN)
A Network with over a 1000 layers? Now we are talking deep Neural Networks. ResNet1
was the first architecture that achieved that while still having a good gradient signal (no
vanishing gradient problem). In their paper they trained a classifier on Cifar10 with 1202
layers. The following figure describes the architecture they used for their CNN (note: the
paper’s main attention is on training a CNN on ImageNet with a similar architecture). ”3x3”
is the kernel size of each feature, where the number after the first comma (e.g. 16) is the
number of output channels of that layer. The last layer of the for loop (the 2n-th ) has a
”/2” indicating a stride of 2. Choose the padding such that the dimensions stay the same
except after applying the stride of 2. This should half the width and height of the input.
The arc arrow defines from where to where the skip connections go and ”fc 10” is a fully
connected layer with a 10-way softmax activation (one output for each of the 10 classes).
Apply batch normalization after each layer.
Build a network with that exact architecture. Find an n that produces a good accuracy.
How deep can you go?
(a) Plot the filters of the first layer. What kind of features do they extract?
(b) For every two convolutions with skip connection calculate the MSE of the input of those
layer xin and the output xout: MSE(xin, xout). Does your network have layers that were
learned to be the identity? (Optionally: Check if this happens during training)
(c) Is deeper always better? Provide some evidence for your answer and explain why that
is the case.

Project11:
Excercise 1
Implement an Autoencoder that encodes the MNIST dataset to a latent dimension of size m < 784.
Use Tranposed Convolutions and/or Unpooling to solve this exercise. Train the Autoencoder and
plot the reconstruction training loss. Plot 5 digits (of your choice) before and after reconstruction.
Do this for two different latent dimension sizes.
Exercise 2
Now that you have built an Autoencoder, it is time to implement a Variational Autoencoder. You
can use the Autoencoder you trained in the previous exercise and adapt it for this exercise. Do not
forget to use the reparametrization trick for sampling from Z-space.
a) Train a Variational Autoencoder with latent dimension of size 2. Then,
plot the digits where their associated position was in latent space similarly as
explained in the lecture

Project12:
Excercise 1. Recurrent Neural Networks (RNNs) (50%)
Fill out the missing code in the given notebook1
that you need to get the RNN to work and
produce some text samples with the trained RNN.
Excercise 2. LSTM and GRU (50%)
Now add the missing code for the LSTM. Instead of an LSTM you can also use GRU here.
Compare both architectures (RNN and the one you choose here) against each other.
