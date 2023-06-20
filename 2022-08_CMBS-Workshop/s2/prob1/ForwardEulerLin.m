function [times, u] = ForwardEulerLin(A, u0, b, tBeg, tEnd, N)
%FORWARDEULERLIN Solve a linear system of ODE using Forward Euler
%    Solve a linear system of ODE using Forward Euler.
%    Considering the problem
%
%    .. math::
%        \\frac{dU}{dt} = AU + b(t),
%
%    computes the numerical solution with Forward Euler, between **tBeg**
%    and **tEnd**, with **u0** the initial solution, using **N** steps.
%
%    Parameters
%    ----------
%    A : matrix of size JxJ
%        The matrix of the linear system
%    u0 : vector of size J
%        The initial vector
%    b : function
%        The source term
%    tBeg : float
%        The initial time of the solution
%    tEnd : float
%        The final time of the solution
%    N : int
%        The number of numerical time steps
% 
%     Returns
%     -------
%     u : matrix of size JxN
%         The solution at each time steps (including initial solution)
%     times : vector of size N
%         The times of the solutions
dt = (tEnd-tBeg)/N;
times = linspace(tBeg, tEnd, N+1);
J = length(u0);
u = zeros(J, N+1);
u(:, 1) = u0;
R = speye(J)+dt*A;
for i=1:N
    u(:, i+1) = R*u(:, i) + dt*b(tBeg+(i-1)*dt);
end
end
