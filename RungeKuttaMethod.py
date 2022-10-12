import numpy as np

rho1 = np.array([1, 1, -1, -1])


def alphamark(beta, k, theta, sigma):

    """ First ODE to solve to obtain A and B for affine models. """

    return (k @ theta).T @ beta + 0.5 * beta.T @ sigma @ sigma.T @ beta


def betamark(k, beta, rho_1):

    """ Second ODE to solve to obtain A and B for affine models. """

    return -rho_1 + k.T @ beta


def RK(funalpha, funbeta, timestep, wa, wb, tau, X):

    """ Runge-Kutta method to solve those ODEs numerically, i.e. giving A and B."""

    alpha = wa
    beta = wb
    obs = int(tau / timestep)

    for ob in range(obs):
        k1 = timestep * funbeta(beta)
        k2 = timestep * funbeta(beta + 0.5 * k1)
        k3 = timestep * funbeta(beta + 0.5 * k2)
        k4 = timestep * funbeta(beta + k3)

        l1 = timestep * funalpha(beta)
        l2 = timestep * funalpha(beta + 0.5 * k1)
        l3 = timestep * funalpha(beta + 0.5 * k2)
        l4 = timestep * funalpha(beta + k3)

        beta = beta + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        alpha = alpha + (1 / 6) * (l1 + 2 * l2 + 2 * l3 + l4)

    return -(alpha + beta.T @ X) / tau
