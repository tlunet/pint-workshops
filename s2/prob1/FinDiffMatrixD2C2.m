function A = FinDiffMatrixD2C2(J,L)
%FinDiffMatrixD2C2 Build a derivation matrix
%   A = FinDiffMatrixD2C2(J,L); Build the derivation matrix for the second
%   derivative, using centered finite difference of order 2 (C2).
%   Parameters: - J, size of the matrix (excluding boundary points)
%               - L, size of the domain (including boundary points)
h = L/(J+1);
A = toeplitz([-2., 1., zeros(1, J-2)]);
A = A/h^2;
end

