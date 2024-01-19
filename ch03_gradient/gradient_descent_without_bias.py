""" 
gradient descent
is a way to find the minimum of the loss function—but it’s faster, more precise than
looping through loss functions
"""
import numpy as np

# This solution does not use bias, setting to 0
b = 0

# Predict Y from X and the slope of the line
def predict(X, w, b):
    return X * w + b

# Calculate the mean squared error to get the loss
def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)

def gradient(X, Y, w):
    return 2 * np.average(X * (predict(X, w, b) - Y))

def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, 0)))
        w -= gradient(X, Y, w) * lr
    return w
 
# Import the dataset
X, Y = np.loadtxt(".venv\ch01_how\pizza.txt", skiprows=1, unpack=True)

# Train the system
w = train(X, Y, iterations=100, lr=0.001)
print("\nw=%.10f" % w)

# Predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))

# Plot the chart
import matplotlib.pyplot as plt
import seaborn as sns

# activate seaborn
sns.set()         

# plot data as blue circles
plt.plot(X, Y, "bo")    

# set x axis ticks                                                                          
plt.xticks(fontsize=15)    

# set y axis ticks                                             
plt.yticks(fontsize=15)    

# set x axis label                                             
plt.xlabel("Reservations", fontsize=30)   

# set y axis label                              
plt.ylabel("Pizzas", fontsize=30)        
 
# set scale axes (0 to 50)
x_edge, y_edge = 50, 50                                                     
plt.axis([0, x_edge, 0, y_edge])   

# plot regression
plt.plot([0, x_edge], [b, predict(x_edge, w, b)], linewidth=1.0, color="g")
    
# display chart                                           
plt.show()                                                              