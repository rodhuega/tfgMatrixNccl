function A = skewlap3d_demo(N)

%   A = SKEWLAP3D_DEMO(N) returns the skew Laplacian matrix of this demo.
%
%   If the numbers 1.5 and 0.5 in this brief program were both equal to 1,
%   it would generate the standard 7-point large sparse discretization of
%   the Laplace operator in three dimensions.  With the numbers 1.5 and 0.5
%   as written, we get a highly nonnormal matrix with the same sparsity
%   pattern.  A matrix like this might arise in the discretization of a
%   convection-diffusion problem.

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  I = speye(N-1);
  D = N^2*toeplitz([-2 1.5 zeros(1,N-3)],[-2 .5 zeros(1,N-3)]);
  A = kron(I,kron(I,D)) + kron(I,kron(D,I)) + kron(D,kron(I,I));

