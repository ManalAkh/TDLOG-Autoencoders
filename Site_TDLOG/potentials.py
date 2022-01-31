import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import matplotlib
matplotlib.use('Agg') # this allows PNG plotting
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def distance(x0,y0,x,y):
    return np.sqrt((x0-x)*(x0-x)+(y-y0)*(y-y0))

def potV(x0,y0,r,x,y,a):
    dis = distance(x0,y0,x,y)
    if dis<=r:
        val = -a*(1-abs(x-x0)/r)*(1-abs(x-x0)/r)-a*(1-abs(y-y0)/r)*(1-abs(y-y0)/r)
        return val
    else:
        return 0



N=1000

class MultimodalPotential:
    def __init__(self,bowls_coord, beta):
        """Initialise potential function class

        :param bowls_coord: matrix, where bowls_coord[i]=([x0,y0,r,a])
        where (x0,y0) are the coords of the center of the bowl and r its radius
        a the intensity of the well of potential made by the bowl
        """
        self.beta = beta
        self._bowls_coord=np.copy(bowls_coord)
        self.dim = 2
        self.nbr_bowls = np.shape(bowls_coord)
        self.Z = None

    def V(self, X):
        """Potential function

        :param X: np.array, Position  vector (x,y), ndim = 1, shape = (2,)
        :return: V: float, potential energy value
        """
        assert(type(X) == np.ndarray)
        assert(X.ndim == 1)
        assert(X.shape[0] == 2)
        x = X[0]
        y = X[1]
        V=0
        bowl=self._bowls_coord
        for bowl in self._bowls_coord:
            x0=bowl[0]
            y0=bowl[1]
            r=bowl[2]
            a=bowl[3]
            dis = np.sqrt((x0-x)*(x0-x)+(y-y0)*(y-y0))
            if dis<r:
                V-=a*(1-(x-x0)*(x-x0)/(r*r))*(1-(y-y0)*(y-y0)/(r*r))
        V+=0.2 * (x ** 4) + 0.2 * ((y - 1/3) ** 4)
        return V
    
    def dV_x(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVx: float, derivative of the potential with respect to x
        """
        dVx = 0
        for bowl in self._bowls_coord:
            x0=bowl[0]
            y0=bowl[1]
            r=bowl[2]
            a=bowl[3]
            dis = np.sqrt((x0-x)*(x0-x)+(y-y0)*(y-y0))
            if dis<r:
                dVx+=2*a*((x-x0)/(r*r))*(1-(y-y0)*(y-y0)/(r*r))
        dVx+= 0.8 * (x**3)
        return dVx
    
    def dV_y(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVy: float, derivative of the potential with respect to y
        """
        dVy = 0
        for bowl in self._bowls_coord:
            x0=bowl[0]
            y0=bowl[1]
            r=bowl[2]
            a=bowl[3]
            dis = np.sqrt((x0-x)*(x0-x)+(y-y0)*(y-y0))
            if dis<r:
                dVy+=2*a*((y-y0)/(r*r))*(1-(x-x0)*(x-x0)/(r*r))

        dVy+= 0.8 * ((y - 1/3)**3)
        return dVy
    
    def nabla_V(self, X):
        """Gradient of the potential energy fuction

        :param X: np.array, Position  vector (x,y), ndim = 1, shape = (2,)
        :return: grad(X): np.array, gradient with respect to position vector (x,y), ndim = 1, shape = (2,)
        """
        assert(type(X) == np.ndarray)
        assert(X.ndim == 1)
        assert(X.shape[0] == 2)
        return np.array([self.dV_x(X[0], X[1]), self.dV_y(X[0], X[1])])
        
    def boltz_weight(self, x, y):
        """Compute the unnormalized weight of a configuration according to the Boltzmann distribution

        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: normalized Blotzmann weight
        """
        X = np.array([x, y])
        return np.exp(-self.beta * self.V(X))
    
    def set_Z(self):
        """Partition function to normalize probability densities
        """
        self.Z, _ = integrate.dblquad(self.boltz_weight, -5, 5, -5, 5)  


def create_plots(Potential):
    """
        creates 3D potential plot
        :param Potential: MultimodalPotential, the potential

        :return: potential figure
        """
    grid = np.linspace(-2,2,100)
    X = np.outer(grid, np.ones(100))
    Y = np.outer(grid + 0.5, np.ones(100)).T
    potential_on_grid = np.zeros([100, 100])
    for i in range(100):
        for j in range(100):
            potential_on_grid[i, j] = Potential.V(np.array([grid[i], grid[j] + 0.5]))

    fig3D = go.Figure(data=[go.Surface(z=potential_on_grid, x=X, y=Y)])
    fig3D.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
    fig3D.update_layout(title='The potential map', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90)       
    )

    return fig3D


def UnbiasedTraj(pot, X_0 = np.array([0, 0]), delta_t=1e-3, T=1000, save=1, save_energy=False, seed=0):
    """Simulates an overdamped langevin trajectory with a Euler-Maruyama numerical scheme 

    :param pot: potential object, must have methods for energy gradient and energy evaluation
    :param X_0: Initial position, must be a 2D vector
    :param delta_t: Discretization time step
    :param T: Number of points in the trajectory (the total simulation time is therefore T * delta_t)
    :param save: Integer giving the period (counted in number of steps) at which the trajectory is saved
    :param save_energy: Boolean parameter to save energy along the trajectory

    :return: traj: np.array with ndim = 2 and shape = (T // save + 1, 2)
    :return: Pot_values: np.array with ndim = 2 and shape = (T // save + 1, 1)
    """
    r = np.random.RandomState(seed)
    X = X_0
    dim = X.shape[0]
    traj = [X]
    if save_energy:
        Pot_values = [pot.V(X)]
    else:
        Pot_values = None
    for i in range(T):
        b = r.normal(size=(dim,))
        X = X - pot.nabla_V(X) * delta_t + np.sqrt(2 * delta_t/pot.beta) * b
        if i % save==0:
            traj.append(X)
            if save_energy:
                Pot_values.append(pot.V(X))
    return np.array(traj), np.array(Pot_values)

def plot_trajectory(Potential, trajectory):
    """
        creates trajectory plot
        :param Potential: MultimodalPotential, the potential
        :param trajectory: np.array with ndim = 2 and shape = (T // save + 1, 2)

        :return: trajectory figure
        """
    #plot the potential
    grid = np.linspace(-2,2,100)
    x_plot = np.outer(grid, np.ones(100))
    y_plot = np.outer(grid + 0.5, np.ones(100)).T
    potential_on_grid = np.zeros([100, 100])
    for i in range(100):
        for j in range(100):
            potential_on_grid[i, j] = Potential.V(np.array([grid[i], grid[j] + 0.5]))

    #plot the trajectory
    fig = plt.figure(figsize=(9,3))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)
    ax0.pcolormesh(x_plot,y_plot,  potential_on_grid, cmap='coolwarm_r', shading='auto')
    ax0.scatter(trajectory[:,0], trajectory[:,1], marker='x')
    ax0.set_xlabel("x coodinate")
    ax0.set_ylabel("y coodinate")
    ax0.set_title("trajectory of the molecule")
    ax1.plot(range(len(trajectory[:,0])), trajectory[:,0], label='x coodinate along trajectory')
    ax1.set_xlabel("time")
    ax1.set_ylabel("x coordinate")
    ax1.set_title('x coordinate along trajectory at time t')
    return fig

