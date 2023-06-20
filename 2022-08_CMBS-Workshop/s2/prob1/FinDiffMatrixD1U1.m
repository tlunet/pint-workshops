function A = FinDiffMatrixD1U1(J,L)
%FinDiffMatrixD1U1 Build a derivation matrix
%   A = FinDiffMatrixD2C2(J,L); Build the derivation matrix for the first
%   derivative, using first order Upwind finite differences and
%   periodic boundary condition
%   Parameters: - J, size of the matrix (excluding right points)
%               - L, size of the domain (including right points)
h = L/J;
v = [1., zeros(1, J-2), -1.];
A = toeplitz([v(1) fliplr(v(2:end))], v);
A = A/h^2;
end
