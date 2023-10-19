#-------------------------------------------------------------------------------
# Equation of a line that passes through the origin is y = x * w or 4x - 3y = 12
#-------------------------------------------------------------------------------
import numpy as np



# Predict Y from X and the slope of the line
def predict(X, w):
    return X * w

# Calculate the mean squared error to get the loss
def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

# Calculate w, the slope of the regression line
def train(X, Y, iterations, lr):
    w=0
    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))
        
        if loss(X, Y, w + lr) < current_loss:
            w += lr
        elif loss(X, Y, w - lr) < current_loss:
            w -= lr
        else:
            return w
        
    raise Exception("Couldn't converge with %d iterations" % iterations)

# Import the dataset
X, Y = np.loadtxt(".venv\ch01_how\pizza.txt", skiprows=1, unpack=True)

# Train the system
w = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f" % w)

# Predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w)))

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
plt.plot([0, x_edge], [0, predict(x_edge, w)], linewidth=1.0, color="g")
    
# display chart                                           
plt.show()                                                              