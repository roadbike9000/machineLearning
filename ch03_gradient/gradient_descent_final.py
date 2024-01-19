""" 
gradient descent
is a way to find the minimum of the loss function—but it’s faster, more precise than
looping through loss functions
"""
import numpy as np

# Predict Y from X and the slope of the line
def predict(X, w, b):
    return X * w + b

# Calculate the mean squared error to get the loss
def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)

# Descent gradient for weight and bias
def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X, w, b) - Y)
    return (w_gradient, b_gradient)

def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, b)))
        w_gradient, b_gradient = gradient(X, Y, w, b)
        w -= w_gradient * lr
        b -= b_gradient * lr
    return w, b
 
# Import the dataset
X, Y = np.loadtxt(".venv\ch01_how\pizza.txt", skiprows=1, unpack=True)

# Train the system
w, b = train(X, Y, iterations=20000, lr=0.001)
print("\nw=%.10f, b=%.10f" % (w, b))

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