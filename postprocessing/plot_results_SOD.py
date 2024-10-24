import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from matplotlib.animation import FuncAnimation

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

def read_data(file_name):
    # Read the data from the given file and return as a numpy array
    data = np.loadtxt(file_name)
    return data

def plot_all_in_one(time_index, T):
    # Read the data from the text files
    rhoA = read_data('../output/SOD/rhoA.txt')
    u = read_data('../output/SOD/u.txt')
    rhoEA = read_data('../output/SOD/rhoEA.txt')
    pressure = read_data('../output/SOD/pressure.txt')

    # Define the x-coordinates (space coordinates)
    x_coords = rhoA[0,1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= rhoA.shape[0]:
        raise ValueError("Invalid time index.")

    rhoA_t = rhoA[time_index, 1:]
    u_t = u[time_index, 1:]
    rhoEA_t = rhoEA[time_index,1:]
    pressure_t = pressure[time_index, 1:]

    # Calculation of analytical solution 
    # Physics
    gg = 1.4  # gamma = C_v / C_p = 7/5 for idrhoEAl gas
    rL, uL, pL =  4,  0.0, 4
    rR, uR, pR = 1, 0.0, 1

    # Set Disretization
    Nx = 500
    X = 1000
    dx = X / (Nx - 1)
    xs = np.linspace(0, X, Nx)
    x0 = Nx // 2

    analytic = SodShockAnalytic(rL, uL, pL, rR, uR, pR, xs, x0, T, gg)
    rhoE_analytic = analytic[2, :] / (gg - 1)  + 1/2*analytic[0, :]*analytic[1, :]**2

    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot Density
    axes[0, 0].plot(x_coords / 1000, rhoA_t / 4, label='MacCormack scheme', color='tab:blue', linewidth=2)
    axes[0, 0].plot(xs / 1000, analytic[0, :] / rL, label="Analytical solution", color="tab:orange", linestyle="--", linewidth=2)
    axes[0, 0].set_xlabel(r'$x/L$', fontsize=12)
    axes[0, 0].set_ylabel(r'Normalized density ($\rho/\rho^{*}$)', fontsize=12)
    axes[0, 0].grid(True)
    axes[0, 0].legend(fontsize=10)

    # Plot Velocity
    axes[0, 1].plot(x_coords / 1000, u_t, label='MacCormack scheme', color='tab:green', linewidth=2)
    axes[0, 1].plot(xs / 1000, analytic[1, :], label="Analytical solution", color="tab:orange", linestyle="--", linewidth=2)
    axes[0, 1].set_xlabel(r'$x/L$', fontsize=12)
    axes[0, 1].set_ylabel(r'$x$-velocity ($u$)', fontsize=12)
    axes[0, 1].grid(True)
    axes[0, 1].legend(fontsize=10)

    # Plot Energy
    axes[1, 0].plot(x_coords / 1000, rhoEA_t / 10, label='MacCormack scheme', color="tab:red", linewidth=2)
    axes[1, 0].plot(xs / 1000, rhoE_analytic / 10, label="Analytical solution", color="tab:orange", linestyle="--", linewidth=2)
    axes[1, 0].set_xlabel(r'$x/L$', fontsize=12)
    axes[1, 0].set_ylabel(r'Normalized total energy ($\frac{\rho E}{\rho^* E^*}$)', fontsize=12)
    axes[1, 0].grid(True)
    axes[1, 0].legend(fontsize=10)

    # Plot Pressure
    axes[1, 1].plot(x_coords / 1000, pressure_t / 4, label='MacCormack scheme', color="tab:purple", linewidth=2)
    axes[1, 1].plot(xs / 1000, analytic[2, :] / pL, label="Analytical solution", color="tab:orange", linestyle="--", linewidth=2)
    axes[1, 1].set_xlabel(r'$x/L$', fontsize=12)
    axes[1, 1].set_ylabel(r'Normalized pressure ($p/p^{*}$)', fontsize=12)
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
    rhoA = read_data('../output/SOD/rhoA.txt')
    u = read_data('../output/SOD/u.txt')
    rhoEA = read_data('../output/SOD/rhoEA.txt')
    pressure = read_data('../output/SOD/pressure.txt')

    # Define the x-coordinates (space coordinates)
    x_coords = rhoA[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= rhoA.shape[0]:
        raise ValueError("Invalid time index.")

    rhoA_t = rhoA[time_index, 1:]
    u_t = u[time_index, 1:]
    rhoEA_t = rhoEA[time_index, 1:]
    pressure_t = pressure[time_index, 1:]

    # Plot Density
    axes[0, 0].plot(x_coords / 1000, rhoA_t / 4, label='MacCormack scheme', color='tab:blue', linewidth=2)

    # Plot Velocity
    axes[0, 1].plot(x_coords / 1000, u_t, label='MacCormack scheme', color='tab:green', linewidth=2)

    # Plot Energy
    axes[1, 0].plot(x_coords / 1000, rhoEA_t / 10, label='MacCormack scheme', color="tab:red", linewidth=2)

    # Plot Pressure
    axes[1, 1].plot(x_coords / 1000, pressure_t / 4, label='MacCormack scheme', color="tab:purple", linewidth=2)

    # Set labels and legends for each subplot
    axes[0, 0].set_xlabel(r'$x/L$', fontsize=12)
    axes[0, 0].set_ylabel(r'Normalized density ($\rho/\rho^{*}$)', fontsize=12)
    axes[0, 0].grid(True)
    axes[0, 0].legend(fontsize=10)

    axes[0, 1].set_xlabel(r'$x/L$', fontsize=12)
    axes[0, 1].set_ylabel(r'$x$-velocity ($u$)', fontsize=12)
    axes[0, 1].grid(True)
    axes[0, 1].legend(fontsize=10)

    axes[1, 0].set_xlabel(r'$x/L$', fontsize=12)
    axes[1, 0].set_ylabel(r'Normalized total energy ($\frac{\rho E}{\rho^* E^*}$)', fontsize=12)
    axes[1, 0].grid(True)
    axes[1, 0].legend(fontsize=10)

    axes[1, 1].set_xlabel(r'$x/L$', fontsize=12)
    axes[1, 1].set_ylabel(r'Normalized pressure ($p/p^{*}$)', fontsize=12)
    axes[1, 1].grid(True)
    axes[1, 1].legend(fontsize=10)

    fig.suptitle(f'SOD shock tube at t = {rhoA[time_index][0]:.2f} s', fontsize=16)

def plot_density(time_index, T):
    # Read the data from the text file
    rhoA = read_data('../output/SOD/rhoA.txt')

    # Define the x-coordinates (space coordinates)
    x_coords = rhoA[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= rhoA.shape[0]:
        raise ValueError("Invalid time index.")

    rhoA_t = rhoA[time_index, 1:]

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

    analytic = SodShockAnalytic(rL, uL, pL, rR, uR, pR, xs, x0, T, gg)


    # Create a plot for density
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords/1000, rhoA_t/4, label='MacCormack scheme', color='tab:blue', linewidth=2)
    plt.plot(xs / 1000, analytic[0, :] / rL, label="Analytical solution", color="tab:orange", linestyle="--", linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'Normalized density ($\rho/\rho^{*}$)', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis = 'both', labelsize = 15)
    plt.show()

def plot_velocity(time_index, T):
    # Read the data from the text file
    u = read_data('../output/SOD/u.txt')

    # Define the x-coordinates (space coordinates)
    x_coords = u[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= u.shape[0]:
        raise ValueError("Invalid time index.")

    u_t = u[time_index, 1:]

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

    analytic = SodShockAnalytic(rL, uL, pL, rR, uR, pR, xs, x0, T, gg)

    # Create a plot for density
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords/1000, u_t, label='MacCormack scheme', color='tab:green', linewidth=2)
    plt.plot(xs / 1000, analytic[1, :], label="Analytical solution", color="tab:orange", linestyle="--", linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'$x$-velocity ($u$)', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis = 'both', labelsize = 15)
    plt.show()

def plot_energy(time_index, T):
    # Read the data from the text file
    rhoEA = read_data('../output/SOD/rhoEA.txt')

    # Define the x-coordinates (space coordinates)
    x_coords = rhoEA[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= rhoEA.shape[0]:
        raise ValueError("Invalid time index.")

    rhoEA_t = rhoEA[time_index, 1:]
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

    analytic = SodShockAnalytic(rL, uL, pL, rR, uR, pR, xs, x0, T, gg)
    rhoE_analytic = analytic[2, :] / (gg - 1)  + 1/2*analytic[0, :]*analytic[1, :]**2


    # Create a plot for density
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords/1000, rhoEA_t/10, label='MacCormack scheme', color="tab:red", linewidth=2)
    plt.plot(xs / 1000, rhoE_analytic/10, label="Analytical solution", color="tab:orange", linestyle="--", linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'Normalized total energy ($\frac{\rho E}{\rho^* E^*}$)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis = 'both', labelsize = 15)
    plt.show()

def plot_pressure(time_index, T):
    # Read the data from the text file
    pressure = read_data('../output/SOD/pressure.txt')

    # Define the x-coordinates (space coordinates)
    x_coords = pressure[0, 1:]  # First row contains the coordinates

    # Select the specified time index for plotting
    if time_index < 0 or time_index >= pressure.shape[0]:
        raise ValueError("Invalid time index.")

    pressure_t = pressure[time_index, 1:]

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

    analytic = SodShockAnalytic(rL, uL, pL, rR, uR, pR, xs, x0, T, gg)
    

    # Create a plot for energy
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords/1000, pressure_t/4, label='MacCormack scheme', color="tab:purple", linewidth=2)
    plt.plot(xs / 1000, analytic[2, :]/pL, label="Analytical solution", color="tab:orange", linestyle="--", linewidth=2)
    plt.xlabel(r'$x/L$', fontsize=18)
    plt.ylabel(r'Normalized pressure ($p/p^{*}$)', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis = 'both', labelsize = 15)
    plt.show()



if __name__ == "__main__":

     # CreaAte a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ani = FuncAnimation(fig, plot_all_in_one_ani, frames=len(read_data('../output/SOD/pressure.txt')[:, 0]) - 1, repeat=False)
    #plt.tight_layout()
    plt.show()



    time_index = len(read_data('../output/SOD/pressure.txt')[:,0])-1
    T = read_data('../output/SOD/pressure.txt')[time_index][0]
    plot_all_in_one(time_index, T)
    plot_density(time_index, T)
    plot_velocity(time_index, T)
    plot_energy(time_index, T)
    plot_pressure(time_index, T)
