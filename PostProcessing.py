import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Initialize global figure variables for each plot type
figs = {}
axes_all_in_one = None

def read_data(file_name):
    """Read the data from the given file and return as a numpy array."""
    return np.loadtxt(file_name)

def plot_residuals_from_file(file_path, save_path=None):
    """Plot residuals for density, velocity, and energy from a file."""
    data = np.loadtxt(file_path, skiprows=1)
    iterations, rhoA_residuals, rhouA_residuals, rhoEA_residuals = data.T
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rhoA_residuals, label=r"$\rho A$ Residual", color='tab:blue')
    plt.plot(iterations, rhouA_residuals, label=r"$\rho u A$ Residual", color='tab:green')
    plt.plot(iterations, rhoEA_residuals, label=r"$\rho E A$ Residual", linestyle='-.', color='tab:red')
    
    plt.xlabel("Iterations", fontsize=15)
    plt.ylabel("Residuals (log scale)", fontsize=15)
    plt.yscale("log")
    plt.title("Evolution of Residuals Over Iterations", fontsize=15)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_mach_colormap(mach_data_path):
    """Plot Mach number distribution as a colormap within the nozzle."""
    mach_data = read_data(mach_data_path)
    time_index = len(mach_data[:, 0]) - 1
    x_coords = mach_data[0, 1:]
    mach_t = mach_data[time_index, 1:]

    x_nozzle = np.linspace(0, 10, 1000)
    area_NOZZLE = lambda x: 1.398 + 0.347 * np.tanh(0.8 * x - 4)
    area = area_NOZZLE(x_nozzle)

    max_radius = np.sqrt(np.max(area) / np.pi)
    X, Y = np.meshgrid(x_coords, np.linspace(-max_radius, max_radius, 500))
    Mach_grid = np.tile(mach_t, (Y.shape[0], 1))
    
    for i, x in enumerate(x_coords):
        radius = np.sqrt(area_NOZZLE(x) / np.pi)
        mask = (Y[:, i] > radius) | (Y[:, i] < -radius)
        Mach_grid[mask, i] = np.nan

    plt.figure(figsize=(10, 6))
    plt.contourf(X / 10, Y / 5, Mach_grid, levels=50, cmap='coolwarm')
    plt.plot(x_nozzle / 10, np.sqrt(area / np.pi) / 5, color='black', linewidth=2)
    plt.plot(x_nozzle / 10, -np.sqrt(area / np.pi) / 5, color='black', linewidth=2)
    
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'$2r/L$', fontsize=18)
    plt.title('Mach distribution in the nozzle', fontsize=20)
    plt.colorbar(label='Mach Number')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def plot_all_in_one(files, xmax, scheme_name):
    """Plot density, velocity, energy, and pressure in one figure with subplots."""
    global axes_all_in_one
    
    rhoA, u, rhoEA, pressure, mach = [read_data(file) for file in files]
    x_coords = rhoA[0, 1:] / xmax
    time_index = len(rhoA) - 1

    if axes_all_in_one is None:
        fig, axes_all_in_one = plt.subplots(2, 2, figsize=(14, 10))
        axes_all_in_one[0, 0].set_ylabel(r'Density ($\rho$)', fontsize=12)
        axes_all_in_one[0, 1].set_ylabel(r'$x$-velocity ($u$)', fontsize=12)
        axes_all_in_one[1, 0].set_ylabel(r'Total energy ($\rho E$)', fontsize=12)
        axes_all_in_one[1, 1].set_ylabel(r'Pressure ($p$)', fontsize=12)
    
    data_list = [rhoA[time_index, 1:], u[time_index, 1:], rhoEA[time_index, 1:], pressure[time_index, 1:]]
    labels = ["Density", "Velocity", "Energy", "Pressure"]
    for ax, data, label in zip(axes_all_in_one.flatten(), data_list, labels):
        ax.plot(x_coords, data, label=scheme_name, linewidth=2)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

def plot_quantity(files, xmax, scheme, file_index, ylabel):
    """Generic function for plotting a single quantity."""
    global figs
    if ylabel not in figs:
        figs[ylabel] = plt.figure(figsize=(10, 6))
    
    data = read_data(files[file_index])
    time_index = len(data) - 1
    x_coords = data[0, 1:] / xmax
    quantity = data[time_index, 1:]

    plt.figure(figs[ylabel].number)
    plt.plot(x_coords, quantity, label=scheme, linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

# Wrapper functions for each variable using plot_quantity
def plot_density(files, xmax, scheme):
    plot_quantity(files, xmax, scheme, 0, r'Density ($\rho A$)')

def plot_velocity(files, xmax, scheme):
    plot_quantity(files, xmax, scheme, 1, r'$x$-velocity ($u$)')

def plot_energy(files, xmax, scheme):
    plot_quantity(files, xmax, scheme, 2, r'Total energy ($\rho E$)')

def plot_pressure(files, xmax, scheme):
    plot_quantity(files, xmax, scheme, 3, r'Pressure ($p$)')

def plot_mach(files, xmax, scheme):
    plot_quantity(files, xmax, scheme, 4, 'Mach Number')
