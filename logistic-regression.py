# In this program, we implement logistic regression with a neural network mindset.
# The goal is to train an algorithm to recognize pictures of cats

import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *

# h5py is a package for interacting with a dataset stored in an .h5 file
# matplotlib is a package for plotting graphs.
# copy is a library that deals with copy and deepcopy. Seems important when objects are constructed recursively.
# In this case it is used to draw pictures, given a (num_px, num_px, 3) array corresponding to the RGB values of their pixels.
# The idea is having a data.h5 dataset containing:
#   - a training set of images of shape (num_px, num_px, 3) labeled as cat (1) or non-cat (0)
#   - a test set of images of shape (num_px, num_px, 3) labeled as cat (1) or non-cat (0)
# We use the training set to train the w and b parameters of logistic regression via gradient descent, by minimizing a cost function
# We use the test set to check how good the trained logistic regression worked
# The dataset is loaded by a couple of commands of the following form:

from lr_utils import load_dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# An example of extracting data from the dataset:

index = 10
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
plt.show()

# With the index specification, the picture is chosen
# plt.imshow shows the picture with number 'index' from train_set_x_orig
# A message telling the corresponding value of y, taken from train_set_y, and whether it is a picture of a cat is printed.
# train_set_y must be a (1,m_train) array of ones and zeros. 
# : typically means 'from 0 to the end' and train_set_y[:, index] extracts the value at position index and puts it into a (,1) array. 
# Had train_set_y been a (n,m_train) array, train_set_y[:, index] would have been (,n) array.
# classes is an array defined in the dataset which associates "non-cat" to '0' and "cat" to '1'.

# As an exercise, we extract the values of m_train, m_test, num_px from the dataset:

m_train= np.size(train_set_y)
m_test= np.size(test_set_y)
num_px = np.shape(train_set_x_orig)[1]

# Note that np.size returns the size of an array, regardless of it being of the (m,1) or (m,) shape.

# We now want to reshape the dataset in such a way that the images are turned into (num_px*num_px*3,1) arrays:

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

# Here we used that, for an array X of shape (a,b,c,d), X.reshape(X.shape[0], -1) has shape (a,b*c*d).
# Obviously, we could put X.reshape(X.shape[1], -1) and obtain a shape (b,a*c*d) shape.
# Here the -1 has the meaning "all the rest", in the sense that Python fills it in with whatever makes most sense.
# The final transpose is there because we want each picture to be a column in the data matrix.

# In machine learning, we often standardize the dataset, which typically means subtracting the overall mean and dividing by standard deviation.
# For picture datasets, where all values go from 0 to 255, typically we just divide by 255:

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# We can now start building the algorithm. We first define the sigmoid function:

def sigmoid(z):
    s = 1 / (1 + np.exp(-z)) 
    return s

# Then we want a function that initializes the two parameters w and b of the logistic regression to zero:

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = np.float(0)
    return w, b

# Note that the two parameters are an (m,1) vertical array w and a float b as wanted.
# To check if some variable you are using is of the correct type, you can use assert. For example: assert type(b) == float

# We can now define a function that from parameters w and b, and data X and Y, defines the cost function and its derivatives

def propagate(w, b, X, Y):
    
    m = X.shape[1]
 
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -(1/m)*(np.dot(Y,np.log(A).T)+np.dot(1-Y,np.log(1-A).T))
  
    dw = (1/m)*np.dot(X,A.T-Y.T)
    db = (1/m)*np.sum(A-Y)
    
    cost = np.squeeze(cost)

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

# Note that the squeeze line acting on cost seems to be pointless, since cost is a scalar. It is probably to ensure it has the right type
# Note also that grads was defined using a Python dictionary.
# Note also that no for-loops were used: np.dot did the trick.

# We are now ready to implement gradient descent.
# This is done with a for-loop, in which we have to specify the number of iterations and the learning rate:

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
        
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
                
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        if i % 100 == 0:
            costs.append(cost)
            
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

# The sintax is pretty straightforward.
# Note that we are also creating a costs list which records the cost computed every 100 iterations.
# i % 100 == 0 indicates i=0 modulo 100.
# Sintax of the form "bla bla bla %i bla %f" %(int, flo)) indicates that %i and %f are an integer and a float to be filled with the values int and flo.
# For the moment ignore the copy.deepcopy lines

# Now we define a function that from parameters w and b predicts the labels Y_prediction for a dataset X:

def predict(w, b, X):
       
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
       
    A = sigmoid(np.dot(w.T,X)+b)
       
    for i in range(A.shape[1]):
        
        if A[0, i] > 0.5 :
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    
    return Y_prediction

# Again, the sintax is very straightforward.

# We now combine all of the above in the model function:

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

# Note that the results are stored in a dictionary.
# The if statement simply computes the accuracies and prints them.
# "bla bla {} bla".format(something) simply puts something instead of the curly brackets.

# Now we can finally apply the model:

logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# It turns out the train accuracy is close to 100%. This should always be the case.
# The test accuracy is 70%, which is ok for a simple model with a small dataset.
# The model is overfitting the training data: this means it follows the training data too closely, and doesn't do well with new test data.

# We can now check the results of our model. For example, we print a picture from the test dataset and the result of our prediction:

index = 3
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(logistic_regression_model['Y_prediction_test'][0,index])].decode("utf-8") +  "\" picture.")
plt.show()

# We can also plot the cost function:

costs = logistic_regression_model["costs"]
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
plt.show()

# The cost is decreasing at each iteration, which means the model is learning. 
# The cost does not reach zero. We can get closer to zero by increasing the number of iterations.
# This will make the training set accuracy go up, but the test set accuracy will go down: overfitting!

# The right choice of learning rate is also important. Too big, you might overshoot the optimal value of the parameters, too small it might take to many iterations to get there.
# We can test the outcome of the model for different learning rates:

learning_rates = [0.01, 0.001, 0.0001]
models = {}

for lr in learning_rates:
    print ("Training a model with learning rate: " + str(lr))
    models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=lr, print_cost=False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for lr in learning_rates:
    plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

# The sintax is straightforward. It's also a good example on how to plot multiple functions with plt.plot and label them.
# The result is that for very small learning rate, the cost decreases very slowly.
# For very high learning rate, the cost oscillates (even though in this specific case it then converges to a small value).

#We can now test the algorithm for an arbitrary image:

my_image = "my_image.jpg"   

# We preprocess the image to fit the algorithm, then apply logistic regression:

fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
plt.show()
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")