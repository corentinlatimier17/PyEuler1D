import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata

def A(x):
    return 1.398 + 0.347*np.tanh(0.8*x-4)

def ratio_A_Athroat(x):
    return A(x)/A(0)

def equation(mach, x):
    gamma = 1.4
    return 1/mach**2*(2/(gamma+1)*(1+(gamma-1)/2*mach**2))**((gamma+1)/(gamma-1))-ratio_A_Athroat(x)**2

def read_data(file_name):
    # Read the data from the given file and return as a numpy array
    data = np.loadtxt(file_name)
    return data

def plot_all_in_one(time_index, T):
    # Read the data from the text files
    rhoA = read_data('../output/Nozzle/rhoA.txt')
    u = read_data('../output/Nozzle/u.txt')
    rhoEA = read_data('../output/Nozzle/rhoEA.txt')
    pressure = read_data('../output/Nozzle/pressure.txt')
    mach = read_data('../output/Nozzle/mach.txt')

    # Define the x-coordinates (space coordinates)
    x_coords = rhoA[0,1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= rhoA.shape[0]:
        raise ValueError("Invalid time index.")

    rhoA_t = rhoA[time_index, 1:]
    u_t = u[time_index, 1:]
    rhoEA_t = rhoEA[time_index,1:]
    pressure_t = pressure[time_index, 1:]
    mach_t = mach[time_index, 1:]

    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot Density
    axes[0, 0].plot(x_coords / 10, rhoA_t, label='MacCormack scheme', color='tab:blue', linewidth=2)
    axes[0, 0].set_xlabel(r'$x/L$', fontsize=12)
    axes[0, 0].set_ylabel(r'Density ($\rho$)', fontsize=12)
    axes[0, 0].grid(True)
    axes[0, 0].legend(fontsize=10)

    # Plot Velocity
    axes[0, 1].plot(x_coords / 10, u_t, label='MacCormack scheme', color='tab:green', linewidth=2)
    axes[0, 1].set_xlabel(r'$x/L$', fontsize=12)
    axes[0, 1].set_ylabel(r'$x$-velocity ($u$)', fontsize=12)
    axes[0, 1].grid(True)
    axes[0, 1].legend(fontsize=10)

    # Plot Energy
    axes[1, 0].plot(x_coords / 10, rhoEA_t, label='MacCormack scheme', color="tab:red", linewidth=2)
    axes[1, 0].set_xlabel(r'$x/L$', fontsize=12)
    axes[1, 0].set_ylabel(r'Total energy ($\rho E$)', fontsize=12)
    axes[1, 0].grid(True)
    axes[1, 0].legend(fontsize=10)

    # Plot Pressure
    axes[1, 1].plot(x_coords / 10, pressure_t, label='MacCormack scheme', color="tab:purple", linewidth=2)
    axes[1, 1].set_xlabel(r'$x/L$', fontsize=12)
    axes[1, 1].set_ylabel(r'Pressure ($p$)', fontsize=12)
    axes[1, 1].grid(True)
    axes[1, 1].legend(fontsize=10)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def plot_all_in_one_ani(time_index):
    # Clear the axes before plotting
    for ax in axes.flatten():  # Flatten the 2D array of axes
        ax.clear()

    # Read the data from the text files
    rhoA = read_data('../output/Nozzle/rhoA.txt')
    u = read_data('../output/Nozzle/u.txt')
    rhoEA = read_data('../output/Nozzle/rhoEA.txt')
    pressure = read_data('../output/Nozzle/pressure.txt')
    mach = read_data('../output/Nozzle/mach.txt')


    # Define the x-coordinates (space coordinates)
    x_coords = rhoA[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= rhoA.shape[0]:
        raise ValueError("Invalid time index.")

    rhoA_t = rhoA[time_index, 1:]
    u_t = u[time_index, 1:]
    rhoEA_t = rhoEA[time_index, 1:]
    pressure_t = pressure[time_index, 1:]
    mach_t = mach[time_index, 1:]

    # Plot Density
    axes[0, 0].plot(x_coords / 10, rhoA_t, label='MacCormack scheme', color='tab:blue', linewidth=2)

    # Plot Velocity
    axes[0, 1].plot(x_coords / 10, u_t, label='MacCormack scheme', color='tab:green', linewidth=2)

    # Plot Energy
    axes[1, 0].plot(x_coords / 10, rhoEA_t, label='MacCormack scheme', color="tab:red", linewidth=2)

    # Plot Pressure
    axes[1, 1].plot(x_coords / 10, pressure_t, label='MacCormack scheme', color="tab:purple", linewidth=2)

    # Set labels and legends for each subplot
    axes[0, 0].set_xlabel(r'$x/L$', fontsize=12)
    axes[0, 0].set_ylabel(r'Density ($\rho$)', fontsize=12)
    axes[0, 0].grid(True)
    axes[0, 0].legend(fontsize=10)

    axes[0, 1].set_xlabel(r'$x/L$', fontsize=12)
    axes[0, 1].set_ylabel(r'$x$-velocity ($u$)', fontsize=12)
    axes[0, 1].grid(True)
    axes[0, 1].legend(fontsize=10)

    axes[1, 0].set_xlabel(r'$x/L$', fontsize=12)
    axes[1, 0].set_ylabel(r'Total energy ($\rho E$)', fontsize=12)
    axes[1, 0].grid(True)
    axes[1, 0].legend(fontsize=10)

    axes[1, 1].set_xlabel(r'$x/L$', fontsize=12)
    axes[1, 1].set_ylabel(r'Pressure ($p$)', fontsize=12)
    axes[1, 1].grid(True)
    axes[1, 1].legend(fontsize=10)

    fig.suptitle(f'Nozzle at t = {rhoA[time_index][0]:.2f} s', fontsize=16)

def plot_density(time_index, T):
    # Read the data from the text file
    rhoA = read_data('../output/Nozzle/rhoA.txt')

    # Define the x-coordinates (space coordinates)
    x_coords = rhoA[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= rhoA.shape[0]:
        raise ValueError("Invalid time index.")

    rhoA_t = rhoA[time_index, 1:]

    # Create a plot for density
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords/10, rhoA_t, label='MacCormack scheme', color='tab:blue', linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'Density ($\rho$)', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis = 'both', labelsize = 15)
    plt.show()

def plot_velocity(time_index, T):
    # Read the data from the text file
    u = read_data('../output/Nozzle/u.txt')

    # Define the x-coordinates (space coordinates)
    x_coords = u[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= u.shape[0]:
        raise ValueError("Invalid time index.")

    u_t = u[time_index, 1:]

    # Create a plot for density
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords/10, u_t, label='MacCormack scheme', color='tab:green', linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'$x$-velocity ($u$)', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis = 'both', labelsize = 15)
    plt.show()

def plot_energy(time_index, T):
    # Read the data from the text file
    rhoEA = read_data('../output/Nozzle/rhoEA.txt')

    # Define the x-coordinates (space coordinates)
    x_coords = rhoEA[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= rhoEA.shape[0]:
        raise ValueError("Invalid time index.")

    rhoEA_t = rhoEA[time_index, 1:]

    # Create a plot for density
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords/10, rhoEA_t, label='MacCormack scheme', color="tab:red", linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'Total energy ($\rho E$)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis = 'both', labelsize = 15)
    plt.show()

def plot_pressure(time_index, T):
    # Read the data from the text file
    pressure = read_data('../output/Nozzle/pressure.txt')

    # Define the x-coordinates (space coordinates)
    x_coords = pressure[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= pressure.shape[0]:
        raise ValueError("Invalid time index.")

    pressure_t = pressure[time_index, 1:]
    

    # Create a plot for pressure
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords/10, pressure_t, label='MacCormack scheme', color="tab:purple", linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'Pressure ($p$)', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis = 'both', labelsize = 15)
    plt.show()


def plot_mach(time_index, T):
    # Read the data from the text file
    mach = read_data('../output/Nozzle/mach.txt')

    # Define the x-coordinates (space coordinates)
    x_coords = mach[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= mach.shape[0]:
        raise ValueError("Invalid time index.")

    mach_t = mach[time_index, 1:]

    x = np.linspace(0, 10, 1000)
    mach_analytic = np.zeros(len(x))
    # Solve for Mach analytically using fsolve
    for i in range(len(x)):
        mach_analytic[i] = fsolve(equation, x0=1.25, args=(x[i],))[0]  # args must be a tuple
    

    # Create a plot for pressure
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords/10, mach_t, label='MacCormack scheme', color="tab:purple", linewidth=2)
    plt.plot(x/10, mach_analytic, label="Analytic solution", color="tab:blue", linewidth=2, linestyle="--")
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'Mach ($M$)', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis = 'both', labelsize = 15)
    plt.show()


def plot_mach_colormap(mach_data_path, time_index):
    # Read the Mach data from the file
    mach_data = read_data(mach_data_path)

    # Extract x-coordinates and Mach numbers for the selected time index
    x_coords = mach_data[0, 1:]  # First row contains x-coordinates
    mach_t = mach_data[time_index, 1:]  # Extract Mach data at given time index

    # Define x for the nozzle shape
    x_nozzle = np.linspace(0, 10, 1000)
    
    # Get the nozzle area for the upper and lower boundaries
    area = A(x_nozzle)

    # Create a meshgrid for the colormap
    max_radius = np.max(np.sqrt(area / np.pi))  # Max nozzle radius
    X, Y = np.meshgrid(x_coords, np.linspace(-max_radius, max_radius, 500))

    # Use the Mach data to create a 2D grid (extend Mach values uniformly in the y-direction)
    Mach_grid = np.tile(mach_t, (Y.shape[0], 1))

    # Mask the values where y > sqrt(A(x)/pi) or y < -sqrt(A(x)/pi)
    for i in range(len(x_coords)):
        area_i = A(x_coords[i])
        radius = np.sqrt(area_i / np.pi)
        mask_upper = Y[:, i] > radius
        mask_lower = Y[:, i] < -radius
        Mach_grid[mask_upper | mask_lower, i] = np.nan  # Mask out-of-bound areas

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot the Mach distribution as a colormap, only within the nozzle
    plt.contourf(X / 10, Y/5 , Mach_grid, levels=50, cmap='coolwarm')

    # Plot the nozzle upper and lower boundaries
    plt.plot(x_nozzle / 10, np.sqrt(area / np.pi)/5, color='black', linewidth=2)
    plt.plot(x_nozzle / 10, -np.sqrt(area / np.pi)/5, color='black', linewidth=2)

    # Add labels, title, and colorbar
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'$2r/L$', fontsize=18)
    plt.title('Mach distribution in the nozzle', fontsize=20)
    plt.colorbar(label='Mach Number')

    # Set equal axis scaling
    plt.axis('equal')

    # Display the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ani = FuncAnimation(fig, plot_all_in_one_ani, frames=len(read_data('../output/Nozzle/pressure.txt')[:, 0]) - 1, repeat=False)
    plt.tight_layout()
    plt.show()
    time_index = len(read_data('../output/Nozzle/pressure.txt')[:,0])-1
    plot_mach_colormap('../output/Nozzle/mach.txt', time_index)
    T = read_data('../output/Nozzle/pressure.txt')[time_index][0]
    plot_mach(time_index, T)

    plot_all_in_one(time_index, T)
    plot_density(time_index, T)
    plot_velocity(time_index, T)
    plot_energy(time_index, T)
    plot_pressure(time_index, T)
