import numpy as np
import matplotlib.pyplot as plt

# define the function f(x,y) = sin(2x)cos(2y) + 0.1(x^2 + y^2)
def z_function(x, y):
    return np.sin(2*x)*np.cos(2*y) + 0.1*(x**2 + y**2)

# compute the gradient (partial derivatives): ∂f/∂x = 2cos(2x)cos(2y) + 0.2x , ∂f/∂y = -2sin(2x)sin(2y) + 0.2y
def calculate_gradient(x, y):
    dx = 2*np.cos(2*x)*np.cos(2*y) + 0.2*x
    dy = -2*np.sin(2*x)*np.sin(2*y) + 0.2*y
    return dx, dy

# generate range of x and y values
x = np.arange(-2, 2, 0.05)
y = np.arange(-2, 2, 0.05)

X, Y = np.meshgrid(x, y) # create a grid of (x, y) values for 3D surface

Z = z_function(X, Y)     # compute Z values for each (x, y) point

current_pos = (0.7, 0.4, z_function(0.7, 0.4)) # initialize starting point and compute its function value

learning_rate = 0.01 # set learning rate (step size)

ax = plt.subplot(projection="3d", computed_zorder=False) # create 3D plot axis

for _ in range(5000): # perform gradient descent iterations
    
    dx, dy = calculate_gradient(current_pos[0], current_pos[1]) # compute gradient at current position
    
    # update x and y using gradient descent rule
    x_new = current_pos[0] - learning_rate * dx
    y_new = current_pos[1] - learning_rate * dy
    
    current_pos = (x_new, y_new, z_function(x_new, y_new)) # update current position with new values
    
    ax.plot_surface(X, Y, Z, cmap="viridis", zorder=0) # plot the surface (cost function)
    
    ax.scatter(current_pos[0], current_pos[1], current_pos[2], color="red", zorder=1) # plot current point as it moves toward the minimum

    # add title and labels
    ax.set_title(r"Gradient Descent on $J(w,b) = \sin(2w)\cos(2b) + 0.1(w^2 + b^2)$")
    ax.set_xlabel("Parameter w")
    ax.set_ylabel("Parameter b")
    ax.set_zlabel("Cost J(w,b)")
    
    # pause briefly to create animation effect
    plt.pause(0.001)
    
    # clear plot for next frame
    ax.clear()



    