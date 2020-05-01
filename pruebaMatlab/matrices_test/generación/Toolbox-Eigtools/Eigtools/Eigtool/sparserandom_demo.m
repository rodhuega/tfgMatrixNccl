function R = sparserandom_demo(N)

%   R = SPARSERANDOM_DEMO(N) returns matrix R of dimension N.
%
%   This is a bidiagonal matrix with exponentially decaying entries on
%   the diagonal and 0.5 on the superdiagonal; a small amount of random
%   noise is then added throughout the matrix. This gives it a  dense
%   spectrum in a ball around the origin with a few well separated
%   eigenvalues. It was used in [1] as an example of a large matrix for
%   which the pseudospectra of the projected matrix created during the 
%   implicitly restarted Arnoldi iteration (ARPACK and eigs) are a good
%   approximation to the pseudospectra of the full matrix. In [1], the
%   dimension N = 200,000 and 30 eigenvalues were requested from a subspace
%   of dimension 50.
%
%   [1]: T. G. Wright and L. N. Trefethen, "Computation of pseudospectra
%        using ARPACK and eigs", SIAM J. Sci. Comp., 23(2), 2001, 591-605

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  R = spdiags([3*exp(-(0:N-1)'/10) .5*ones(N,1)], 0:1, N, N) ...
                     + .1*sprandn(N,N,10/N);
