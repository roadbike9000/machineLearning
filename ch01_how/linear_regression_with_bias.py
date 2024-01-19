#-------------------------------------------------------------------------------
# Equation of a line that passes through the origin is y = x * m or 4x - 3y = 12
# Equation of a line that does not pass through the origin y = m * x + b
# For ML w is used for slope and called "weight", b is called "bias"
#-------------------------------------------------------------------------------
import numpy as np



# Predict Y from X and the slope of the line
def predict(X, w, b):
    return X * w + b

# Calculate the mean squared error to get the loss
def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)

# Calculate w, the slope of the regression line; lr = learning rate or step size
# lr is used to calculate the weight and bias of the regression line
def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %4d => Loss: %.3f" % (i, current_loss))

        if loss(X, Y, w + lr, b) < current_loss:
            w += lr
        elif loss(X, Y, w - lr, b) < current_loss:
            w -= lr
        elif loss(X, Y, w, b + lr) < current_loss:
            b += lr
        elif loss(X, Y, w, b - lr) < current_loss:
            b -= lr
        else:
            return w, b

    raise Exception("Couldn't converge within %d iterations" % iterations)

# Import the dataset
X, Y = np.loadtxt(".venv\ch01_how\pizza.txt", skiprows=1, unpack=True)

# Train the system
w, b = train(X, Y, iterations=100000, lr=0.001)
print("\nw=%.3f, b=%.3f" % (w, b))

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