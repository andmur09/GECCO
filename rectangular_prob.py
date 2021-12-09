from scipy.stats.mvn import mvnun as rectangular
from scipy.stats import multivariate_normal as norm
import numpy as np
import matplotlib.pyplot as plt

# mean = np.array([100, 100])
# cov = np.array([[20, 0], [0, 20]])
# x = mvnun(np.array([95, 95]), np.array([105, 105]), mean, cov)
# print(x)

def gradient(mean, cov, z):
    """
    Description: Finds gradient vector of rectangular probability F(zl, zu) of 
    multivariate normal distribution for given zl and zu

    mean = n * 1 Numpy array of mean vector for random variable
    cov = n * N Numpy array of Covariance Matrix for random variable
    z = 2 * n Numpy array of upper and lower bounds for iteration k
    """
    n = int(np.shape(mean)[0])
    zl, zu = z[0], z[1]
    dzl, dzu = [], []
    xi = norm(mean, cov)

    if n != np.shape(cov[0]) or n != np.shape(cov[1]) or n != np.shape([z]):
        raise AttributeError("Dimension of arrays are not compatible")
    for i in range(n):
        col = cov[:, i].reshape(-1, 1)
        row = cov[:, i].reshape(1, -1)

        bar_mean_l =  mean + 1/cov[i, i] * (zl[i] - mean[i]) * cov[:,i]
        bar_mean_u = mean + 1/cov[i, i] * (zu[i] - mean[i]) * cov[:, i]
        bar_cov = cov - 1/int(cov[i, i])  * col @ row
        bar_zl, bar_zu = np.delete(zl, i), np.delete(zu, i)

        f_u = xi.pdf(zu[i])
        f_l = xi.pdf(zl[i])

        bar_F_u = rectangular(bar_zl, bar_zu, bar_mean_u, bar_cov)[0]
        bar_F_l = rectangular(bar_zl, bar_zu, bar_mean_l, bar_cov)[0]


        dzu.append(f_u * bar_F_u)
        dzl.append(-1* f_l * bar_F_l)
        
    dzl = np.array(dzl)
    dzu = np.array(dzu)

    return zip(dzl, dzu)

def compute_rectangular_2d(lb, ub, mu, cov):
    rv = norm(mu, cov, allow_singular=True)
    Faa = rv.cdf([lb[0], lb[1]])
    Fab = rv.cdf([lb[0], ub[1]])
    Fba = rv.cdf([ub[0], lb[1]])
    Fbb = rv.cdf([ub[0], lb[0]])
    print(Faa, Fab, Fba, Fbb)
    return Faa - Fab - Fba + Fbb


D = 2
x = np.random.rand(D)
mu = np.zeros(D)
# random symmetric matrix
cov = np.array([[1, 0], [0, 1]])

# Generate grid points
x, y = np.meshgrid(np.linspace(-1,2,100),np.linspace(-1,2,100))
xy = np.column_stack([x.flat, y.flat])

# density values at the grid points
Z = norm.pdf(xy, mu, cov).reshape(x.shape)

# arbitrary contour levels
contour_level = [0.1, 0.2, 0.3]

fig = plt.contour(x, y, Z, levels = contour_level)

# print(rectangular(low, up, mu, cov))
# print(compute_rectangular_2d(low, up, mu, cov))
