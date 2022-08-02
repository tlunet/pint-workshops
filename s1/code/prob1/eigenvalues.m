% Script to compute and display the eigenvalues of a matrix
a=31; b=12; c=-27;
mJac = [[a, b, 0];
        [a, 0, c];
        [c, b, a]];
    
vLam = eig(mJac);
nLam = length(vLam);

for i=1:nLam
    lam = vLam(i);
    fprintf('lam%d = %1.2f + %1.2fj\n', i, real(lam), imag(lam))
end

