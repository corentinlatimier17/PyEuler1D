import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton


# Initialize global figure variables for each plot type
figs = {}
axes_all_in_one = None

def read_data(file_name):
    """Read the data from the given file and return as a numpy array."""
    return np.loadtxt(file_name)

def plot_residuals_from_file(file_path,scheme, save_path=None):
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
    plt.title("Evolution of residuals over iterations / " + scheme, fontsize=15)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

def plot_mach_colormap(mach_data_path, scheme):
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
    plt.title('Mach distribution in the nozzle / ' + scheme, fontsize=20)
    plt.colorbar(label='Mach Number')
    plt.axis('equal')
    plt.tight_layout()

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

# Function to find the roots of!
def f(P, pL, pR, cL, cR, gg):
    a = (gg-1)*(cR/cL)*(P-1) 
    b = np.sqrt( 2*gg*(2*gg + (gg+1)*(P-1) ) )
    return P - pL/pR*( 1 - a/b )**(2.*gg/(gg-1.))

# Analtyic Sol to Sod Shock
def SodShockAnalytic(rL, uL, pL, rR, uR, pR, xs, x0, T, gg):
    # rL, uL, pL, rR, uR, pR : Initial conditions of the Reimann problem 
    # xs: position array (e.g. xs = [0,dx,2*dx,...,(Nx-1)*dx])
    # x0: THIS IS AN INDEX! the array index where the interface sits.
    # T: the desired solution time
    # gg: adiabatic constant 1.4=7/5 for a 3D diatomic gas
    dx = xs[1];
    Nx = len(xs)
    v_analytic = np.zeros((3,Nx),dtype='float64')

    # compute speed of sound
    cL = np.sqrt(gg*pL/rL); 
    cR = np.sqrt(gg*pR/rR);
    # compute P
    P = newton(f, 0.5, args=(pL, pR, cL, cR, gg), tol=1e-12);

    # compute region positions right to left
    # region R
    c_shock = uR + cR*np.sqrt( (gg-1+P*(gg+1)) / (2*gg) )
    x_shock = x0 + int(np.floor(c_shock*T/dx))
    v_analytic[0,x_shock-1:] = rR
    v_analytic[1,x_shock-1:] = uR
    v_analytic[2,x_shock-1:] = pR
    
    # region 2
    alpha = (gg+1)/(gg-1)
    c_contact = uL + 2*cL/(gg-1)*( 1-(P*pR/pL)**((gg-1.)/2/gg) )
    x_contact = x0 + int(np.floor(c_contact*T/dx))
    v_analytic[0,x_contact:x_shock-1] = (1 + alpha*P)/(alpha+P)*rR
    v_analytic[1,x_contact:x_shock-1] = c_contact
    v_analytic[2,x_contact:x_shock-1] = P*pR
    
    # region 3
    r3 = rL*(P*pR/pL)**(1/gg);
    p3 = P*pR;
    c_fanright = c_contact - np.sqrt(gg*p3/r3)
    x_fanright = x0 + int(np.ceil(c_fanright*T/dx))
    v_analytic[0,x_fanright:x_contact] = r3;
    v_analytic[1,x_fanright:x_contact] = c_contact;
    v_analytic[2,x_fanright:x_contact] = P*pR;
    
    # region 4
    c_fanleft = -cL
    x_fanleft = x0 + int(np.ceil(c_fanleft*T/dx))
    u4 = 2 / (gg+1) * (cL + (xs[x_fanleft:x_fanright]-xs[x0])/T )
    v_analytic[0,x_fanleft:x_fanright] = rL*(1 - (gg-1)/2.*u4/cL)**(2/(gg-1));
    v_analytic[1,x_fanleft:x_fanright] = u4;
    v_analytic[2,x_fanleft:x_fanright] = pL*(1 - (gg-1)/2.*u4/cL)**(2*gg/(gg-1));

    # region L
    v_analytic[0,:x_fanleft] = rL
    v_analytic[1,:x_fanleft] = uL
    v_analytic[2,:x_fanleft] = pL

    return v_analytic

def plot_quantity(files, xmax, scheme, file_index, ylabel, SOD):
    """Generic function for plotting a single quantity."""
    global figs
    if ylabel not in figs:
        figs[ylabel] = plt.figure(figsize=(10, 6))
   
    data = read_data(files[file_index])
    time_index = len(data) - 1
    x_coords = data[0, 1:] / xmax
    quantity = data[time_index, 1:]

    plt.figure(figs[ylabel].number)
    if SOD is not None:
        # Calculation of analytical solution 
        # Physics
        gg=1.4  # gamma = C_v / C_p = 7/5 for ideal gas
        rL, uL, pL =  4,  0.0, 4
        rR, uR, pR = 1, 0.0, 1
        # Set Disretization
        Nx =500
        X = 1000
        dx = X/(Nx-1)
        xs = np.linspace(0,X,Nx)
        x0 = Nx//2
        T = data[time_index][0]
        analytic = SodShockAnalytic(rL, uL, pL, rR, uR, pR, xs, x0, T, gg)
        if file_index==0:
            plt.plot(xs/xmax, analytic[0, :], label="Analytical solution", linewidth=2, color="tab:purple", linestyle="--")
        elif file_index==1:
            plt.plot(xs/xmax, analytic[1, :], label="Analytical solution", linewidth=2, color="tab:purple", linestyle="--")
        elif file_index==2:
            plt.plot(xs/xmax, analytic[2, :]/(gg-1)+0.5*analytic[0,:]*analytic[1,:]**2, label="Analytical solution", linewidth=2, color="tab:purple", linestyle="--")
        elif file_index==3:
            plt.plot(xs/xmax, analytic[2, :], label="Analytical solution", linewidth=2, color="tab:purple", linestyle="--")
        elif file_index==4:
            plt.plot(xs/xmax, analytic[1, :]/np.sqrt(gg*analytic[2,:]/analytic[0, :]), label="Analytical solution", linewidth=2, color="tab:purple", linestyle="--")
    plt.plot(x_coords, quantity, label=scheme, linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

# Wrapper functions for each variable using plot_quantity
def plot_density(files, xmax, scheme, SOD=None):
    plot_quantity(files, xmax, scheme, 0, r'Density ($\rho A$)', SOD)

def plot_velocity(files, xmax, scheme, SOD=None):
    plot_quantity(files, xmax, scheme, 1, r'$x$-velocity ($u$)', SOD)

def plot_energy(files, xmax, scheme, SOD=None):
    plot_quantity(files, xmax, scheme, 2, r'Total energy ($\rho E$)', SOD)

def plot_pressure(files, xmax, scheme, SOD=None):
    plot_quantity(files, xmax, scheme, 3, r'Pressure ($p$)', SOD)

def plot_mach(files, xmax, scheme, SOD=None):
    plot_quantity(files, xmax, scheme, 4, 'Mach Number', SOD)
