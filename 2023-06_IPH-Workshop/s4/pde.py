import numpy as np
import scipy.linalg as spl

def finDiffMatrixD2C2(J, L):
    """Compute the finite-difference matrix for the second derivative
    in space (D2) using centered finite differences of order 2 (C2).

    Parameters
    ----------
    J : int
        Number of mesh point in space (excluding boundary conditions)
    L : float
        Length of the domain (including boundary conditions)
    """
    h = L/(J+1)
    A = spl.toeplitz([-2., 1.]+(J-2)*[0.])
    A /= h**2
    return A


def finDiffMatrixD1U1(J, L):
    """Compute the finite-difference matrix for the first derivative
    in space (D1) using first order Upwind finite differences (U1),
    and periodic boundary conditions.

    Parameters
    ----------
    J : int
        Number of mesh point in space (excluding right point)
    L : float
        Length of the domain (including right point)
    """
    h = L/J
    A = spl.circulant([1, -1.]+(J-2)*[0.])
    A /= h
    return A


def finDiffMatrixD1C2(J, L):
    """Compute the finite-difference matrix for the first derivative
    in space (D1) using second order centered finite differences (C2),
    and periodic boundary conditions.

    Parameters
    ----------
    J : int
        Number of mesh point in space (excluding right point)
    L : float
        Length of the domain (including right point)
    """
    h = L/J
    A = spl.circulant([0, -1.]+(J-3)*[0.] + [1.])
    A /= 2*h
    return A


def backwardEulerLin(A, u0, b, tBeg, tEnd, N):
    """
    Solve a linear system of ODE using Backward Euler.
    Considering the problem

    .. math::
        \\frac{dU}{dt} = AU + b(t),

    computes the numerical solution with Backward Euler, between **tBeg**
    and **tEnd**, with **u0** the initial solution, using **N** steps.

    Parameters
    ----------
    A : matrix of size JxJ
        The matrix of the linear system
    u0 : vector of size J
        The initial solution
    b : function
        The source term
    tBeg : float
        The initial time of the solution
    tEnd : float
        The final time of the solution
    N : int
        The number of numerical time steps

    Returns
    -------
    u : matrix of size JxN
        The solution at each time steps (including initial solution)
    times : vector of size N
        The times of the solutions
    """
    dt = (tEnd-tBeg)/N
    times = np.linspace(tBeg, tEnd, N+1)
    u0 = np.asarray(u0)
    J = u0.size
    u = np.zeros((J, N+1), dtype=u0.dtype, order='F')
    u[:, 0] = u0
    R = np.copy(A)
    R *= -dt
    R += np.eye(J)
    for i, t in enumerate(times[1:]):
        u[:, i+1] = np.linalg.solve(R, u[:, i] + dt*b(t))
    return times, u


def parareal(F, G, T, u0, N, K):
    """
    Compute the solutions from the Parareal algorithm on a given global
    time interval.

    Parameters
    ----------
    F : function(t0, t1, u0) -> u1
        Function returning the fine solution.
    G : function(t0, t1, u0) -> u1
        Function returning the coarse solution.
    T : float
        Length of the global time interval.
    u0 : vector
        The initial value for the algorithm.
    N : int
        The number of time sub-intervals.
    K : int
        The number of iteration.

    Returns
    -------
    u : np.array(K+1, N+1, size(u0))
        The Parareal solution at each iterations and each time sub-interval.
    """
    t = np.linspace(0, T, N+1)
    u0 = np.asarray(u0)
    u = np.zeros((K+1, N+1, u0.size))
    
    # Initial solution for each k
    u[:, 0] = u0

    # Initial coarse propagation
    for n in range(N):
        u[0, n+1] = G(t[n], t[n+1], u[0, n])

    # Iterations
    for k in range(K):
        for n in range(N):
            uF = F(t[n], t[n+1], u[k, n])
            uGk1 = G(t[n], t[n+1], u[k+1, n])
            uGk = G(t[n], t[n+1], u[k, n])
            u[k+1, n+1] = uF + uGk1 - uGk
    
    return u

