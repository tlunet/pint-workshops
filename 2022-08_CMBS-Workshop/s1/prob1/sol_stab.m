% Set Lorenz system parameters
sigma = 10; rho = 28; beta = 8/3;

K = (beta*(rho-1))^0.5;

% Build the Jacobian matrix at one fixed point
mJac = [[-sigma, sigma, 0];
        [1, -1, -K];
        [K, K, -beta]];

% Compute the eigenvalues
vLam = eig(mJac);
nLam = length(vLam);

% Print the eigenvalues
for i = 1:nLam
    lam = vLam(i);
    fprintf('lam%d = %1.2f + %1.2fj\n', i, real(lam), imag(lam));
end