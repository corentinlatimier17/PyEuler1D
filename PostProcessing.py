import numpy as np
import matplotlib.pyplot as plt

def plot_residuals_from_file(file_path, save_path=None):
    """
    Reads residuals from a .txt file and plots their evolution over iterations.

    Parameters:
    - file_path (str): Path to the .txt file containing residuals.
    - save_path (str, optional): Path to save the plot. If None, the plot is shown interactively.
    """
    # Load data from file
    data = np.loadtxt(file_path, skiprows=1)  # skip the header row

    # Extract columns
    iterations = data[:, 0]
    rhoA_residuals = data[:, 1]
    rhouA_residuals = data[:, 2]
    rhoEA_residuals = data[:, 3]

    # Plot residuals for each variable
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rhoA_residuals, label=r"$\rho A$ Residual", linestyle='-', color='tab:blue')
    plt.plot(iterations, rhouA_residuals, label=r"$\rho u A$ Residual", linestyle='-', color='tab:green')
    plt.plot(iterations, rhoEA_residuals, label=r"$\rho E A$ Residual", linestyle='-.', color='tab:red')
    
    # Labeling and grid
    plt.xlabel("Iterations", fontsize=15)
    plt.ylabel("Residuals (log scale)", fontsize=15)
    plt.yscale("log")  # Log scale for better convergence visualization
    plt.title("Evolution of Residuals Over Iterations", fontsize=15)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    # Show or save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

def read_data(file_name):
    # Read the data from the given file and return as a numpy array
    data = np.loadtxt(file_name)
    return data

def plot_mach_colormap(mach_data_path):
    # Read the Mach data from the file
    mach_data = read_data(mach_data_path)
    time_index = len(mach_data[:,0])-1

    # Extract x-coordinates and Mach numbers for the selected time index
    x_coords = mach_data[0, 1:]  # First row contains x-coordinates
    mach_t = mach_data[time_index, 1:]  # Extract Mach data at given time index

    # Define x for the nozzle shape
    x_nozzle = np.linspace(0, 10, 1000)
    
    # Get the nozzle area for the upper and lower boundaries

    def area_NOZZLE(x):
        return 1.398 + 0.347*np.tanh(0.8*x-4)
    area = area_NOZZLE(x_nozzle)

    # Create a meshgrid for the colormap
    max_radius = np.max(np.sqrt(area / np.pi))  # Max nozzle radius
    X, Y = np.meshgrid(x_coords, np.linspace(-max_radius, max_radius, 500))

    # Use the Mach data to create a 2D grid (extend Mach values uniformly in the y-direction)
    Mach_grid = np.tile(mach_t, (Y.shape[0], 1))

    # Mask the values where y > sqrt(A(x)/pi) or y < -sqrt(A(x)/pi)
    for i in range(len(x_coords)):
        area_i = area_NOZZLE(x_coords[i])
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

def plot_all_in_one(files, xmax, scheme_name):
    # Read the data from the text files
    rhoA = read_data(files[0])
    u = read_data(files[1])
    rhoEA = read_data(files[2])
    pressure = read_data(files[3])

    time_index = len(rhoA[:,0])-1

    # Define the x-coordinates (space coordinates)
    x_coords = rhoA[0,1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= rhoA.shape[0]:
        raise ValueError("Invalid time index.")

    rhoA_t = rhoA[time_index, 1:]
    u_t = u[time_index, 1:]
    rhoEA_t = rhoEA[time_index,1:]
    pressure_t = pressure[time_index, 1:]

    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot Density
    axes[0, 0].plot(x_coords / xmax, rhoA_t, label=scheme_name + ' scheme', color='tab:blue', linewidth=2)
    axes[0, 0].set_xlabel(r'$x/L$', fontsize=12)
    axes[0, 0].set_ylabel(r'Density ($\rho$)', fontsize=12)
    axes[0, 0].grid(True)
    axes[0, 0].legend(fontsize=10)

    # Plot Velocity
    axes[0, 1].plot(x_coords / xmax, u_t, label=scheme_name + ' scheme', color='tab:green', linewidth=2)
    axes[0, 1].set_xlabel(r'$x/L$', fontsize=12)
    axes[0, 1].set_ylabel(r'$x$-velocity ($u$)', fontsize=12)
    axes[0, 1].grid(True)
    axes[0, 1].legend(fontsize=10)

    # Plot Energy
    axes[1, 0].plot(x_coords / xmax, rhoEA_t, label=scheme_name + ' scheme', color="tab:red", linewidth=2)
    axes[1, 0].set_xlabel(r'$x/L$', fontsize=12)
    axes[1, 0].set_ylabel(r'Total energy ($\rho E$)', fontsize=12)
    axes[1, 0].grid(True)
    axes[1, 0].legend(fontsize=10)

    # Plot Pressure
    axes[1, 1].plot(x_coords / xmax, pressure_t, label=scheme_name + ' scheme', color="tab:purple", linewidth=2)
    axes[1, 1].set_xlabel(r'$x/L$', fontsize=12)
    axes[1, 1].set_ylabel(r'Pressure ($p$)', fontsize=12)
    axes[1, 1].grid(True)
    axes[1, 1].legend(fontsize=10)

    # Adjust layout and show the plot
    plt.tight_layout()

def plot_density(files, xmax, scheme):
    # Read the data from the text file
    rhoA = read_data(files[0])
    time_index = len(rhoA[:,0])-1
    # Define the x-coordinates (space coordinates)
    x_coords = rhoA[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= rhoA.shape[0]:
        raise ValueError("Invalid time index.")

    rhoA_t = rhoA[time_index, 1:]

    # Create a plot for density
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords/xmax, rhoA_t, label= scheme + ' scheme', color='tab:blue', linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'Density ($\rho A$)', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis = 'both', labelsize = 15)

def plot_velocity(files, xmax, scheme):
     # Read the data from the text file
    u = read_data(files[1])
    time_index = len(u[:,0])-1

    # Define the x-coordinates (space coordinates)
    x_coords = u[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= u.shape[0]:
        raise ValueError("Invalid time index.")

    u_t = u[time_index, 1:]

    # Create a plot for u
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords/xmax, u_t, label=scheme + ' scheme', color='tab:green', linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'$x$-velocity ($u$)', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis = 'both', labelsize = 15)

def plot_energy(files, xmax, scheme):
     # Read the data from the text file
    rhoEA = read_data(files[2])
    time_index = len(rhoEA[:,0])-1

    # Define the x-coordinates (space coordinates)
    x_coords = rhoEA[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= rhoEA.shape[0]:
        raise ValueError("Invalid time index.")

    rhoEA_t = rhoEA[time_index, 1:]

    # Create a plot for u
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords/xmax, rhoEA_t, label=scheme + ' scheme', color='tab:green', linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'Total energy ($\rho EA$)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis = 'both', labelsize = 15)

def plot_pressure(files, xmax, scheme):
     # Read the data from the text file
    p = read_data(files[3])
    time_index = len(p[:,0])-1

    # Define the x-coordinates (space coordinates)
    x_coords = p[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= p.shape[0]:
        raise ValueError("Invalid time index.")

    p_t = p[time_index, 1:]

    # Create a plot for u
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords/xmax, p_t, label=scheme + ' scheme', color='tab:green', linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'Pressure ($p$)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis = 'both', labelsize = 15)

def plot_mach(files, xmax, scheme):
     # Read the data from the text file
    mach = read_data(files[4])
    time_index = len(mach[:,0])-1

    # Define the x-coordinates (space coordinates)
    x_coords = mach[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= mach.shape[0]:
        raise ValueError("Invalid time index.")

    mach_t = mach[time_index, 1:]

    # Create a plot for u
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords/xmax, mach_t, label=scheme + ' scheme', color='tab:green', linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'Mach number', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis = 'both', labelsize = 15)
