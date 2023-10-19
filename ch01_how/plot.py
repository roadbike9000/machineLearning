import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# activate seaborn
sns.set()         
# scale axes (0 to 50)                                                       
plt.axis([0, 50, 0, 50])   
# set x axis ticks                                                                          
plt.xticks(fontsize=15)    
# set y axis ticks                                             
plt.yticks(fontsize=15)    
# set x axis label                                             
plt.xlabel("Reservations", fontsize=30)   
# set y axis label                              
plt.ylabel("Pizzas", fontsize=30)        
# load data                               
X, Y = np.loadtxt(".venv\ch01_how\pizza.txt", skiprows=1, unpack=True)  
 # plot data as blue circles
plt.plot(X, Y, "bo")        
# display chart                                           
plt.show()                                                              