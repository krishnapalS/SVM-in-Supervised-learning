# Chandler Severson
# Support Vector Machines - Machine Learning
# Jul 22, 2017

# Support Vector Machines
#   -Used for Classification, regression (time serires production, etc),
#    outlier detection, clustering
#
#   -SVMs are great if you have small datasets (e.g. 1000 rows or fewer)
#   -Other algorithms (Random forests, deep neural networks, etc..)
#       require more data but come up with very robust models

# What is a SVM?
#   -Supervised ML Algorithm that Can do Regression and Classification.
#   -Regression: "Find the next point in a series"
#   -Classification: "Is it this, this, or that?"
#       -Given 2+ labeled classes of data (Supervised Learning),
#        it acts as a discriminative (opposite is generative)
#        classifier. Formally defined by an optimal hyperplane that
#        seperates all of the classes.
#
#   -Hyperplane: Decision boundary between classes.
#       -Built: maximizing margin(space) between the line and
#       both of the classes.
#       -Support Vectors: Points in each CLASS that are closest
#       to the Decision Boundary. These points SUPPORT the creation
#       of the Hyperplane from our SVM.
#
#   -Hyperplane: A linear decision surface that splits the space
#   into two parts. (N-1 dimensions). Hyperplanes are binary
#   classifiers.

# How do they Classify?
#   -A line is drawn between the classes in the middle of the
#    Hyperplane.
#   -Depending on where a new data point is in relation to this line
#    Classifies what the point is.

# What are Learning Rates?
#   -The length of the steps the algorithm makes down the gradient
#    on the error curve.
#   -If too HIGH, the algorithm might overshoot the optimal point
#   -If too LOW, the algorithm may take too long to converge or NEVER converge.
#

# Loss function (http://bit.ly/2um9lIX)
#   *We will use Hinge Loss: a popular algorithm for SVMs.
#       -Used for Maximum-Margin classification
#
#   -Hinge Loss: c(x,y,f(x)) = (1 - y * f(x))
#       * c: Hinge Loss - Must be positive or Zero (http://bit.ly/2vxbyjs).
#       * x: Sample Data
#       * y: True Label
#       * f(x): Predicted Label
#

# Objective Function (http://bit.ly/2uLEu9Q)
#   -Objective of SVM consists of two terms:
#       -a REGULARIZER & LOSS FUNCTION
# REGULARIZER: Balances between margin maximization and loss.
#    Finds the decision surface that is maximally far away from
#    any data points.
#   -Tells us how best to fit our data. Controls the trade-off between
#    a low training error and a low testing error
#   -Term is 1/epochs. This parameter will decrease as epochs increases.
#
#
#   -REGULARIZER TOO HIGH: model is OVERFIT to training data.
#   -OVERFITTING: When model is tuned to data such that it will
#    have a hard time generalizing new entries/data points
#
#   -REGULARIZER TOO LOW: model is UNDERFIT to training data.
#   -UNDERFITTING: When model is less tuned to data such that it
#   will be TOO general.
#
#   -PERFECT REGULARIZER TERM: model is general enough while being
#    fit to our data. Perfect balance between over/underfitting
#
#   -Takes sum of Loss from all data points. Represents Total Loss
#   -This loss is added to the REGULARIZER.
#
# How to Optimize?
#   -Find the optimal REGULARIZER term.
#   -Minimize the loss.
#
#   -Perform Gradient Descent on Objective Function
#       -http://bit.ly/2eEBEg5
#       -Take Partial Derivative of both Regularizer AND Loss
#   -If we have a misclassified sample, we update the weight vector W
#    using the gradients of both terms, else if classified correctly,
#    we just update W by the gradient of the regularizer
#   -Misclassified IF y<x,w> < 1.
#   -Update rule: w = w+n(y*x - 2lw)
#       -(n is learning rate, l is regularizer)

# For Matrix Math Operations
import numpy as np

# To plot out our data
from matplotlib import pyplot as plt

# Stochastic Gradient Descent.
def svm(x,y):
    learning_rate   = 1

    # Initialize our SVMs weight vector with all zeros (3 values)
    w               = np.zeros(len(x[0]))

    # How many iterations to train for
    epochs          = 100000

    # Store misclassifications so we can plot them later
    errors          = []

    # Train and Gradient Descent
    for epoch in range(1,epochs):
        error = 0
        for i,n in enumerate(x):
            #misclassification
            if(y[i] * np.dot(x[i], w)) < 1:
                #missclassified weight update
                w = w + learning_rate * ( (x[i] * y[i]) + (-2 * (1/epoch) * w) )
                error = 1
            else:
                #correct classified weight update
                w = w + learning_rate * (-2 * (1/epoch) * w)
        errors.append(error)

    # Plot the rate of classification errors during training
    plt.plot(errors, '|')
    plt.ylim(0.5, 1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()

    return w


def run():
    # Input data - [X value, Y value, Bias term]
    X = np.array([
        [-2,4,-1],
        [4,1,-1],
        [1,6,-1],
        [2,4,-1],
        [6,2,-1],
    ])

    # Associated output labels
    Y = np.array([-1,-1,1,1,1])

    # Run the algorithm on the sample data above
    w = svm (X,Y)

    # Plot the output
    for d, sample in enumerate(X):
        # Plot the negative samples
        if d < 2:
            plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
        # Plot the positive samples
        else:
            plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

    # Add our test samples
    plt.scatter(2,2, s=120, marker='_', linewidths=2, color='yellow')
    plt.scatter(4,3, s=120, marker='+', linewidths=2, color='blue')

    # Print the hyperplane calculated by svm()
    x2=[w[0],w[1],-w[1],w[0]]
    x3=[w[0],w[1],w[1],-w[0]]

    x2x3 =np.array([x2,x3])
    X,Y,U,V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X,Y,U,V,scale=1, color='blue')
    plt.show()



if __name__ == '__main__':
    run()
