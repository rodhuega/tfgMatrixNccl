function G = grcar_demo(N)

%   G = GRCAR_DEMO(N) returns matrix G of dimension N.
%
%   The Grcar matrix is also in the MATLAB gallery, and can be
%   created by the command G = gallery('grcar',N)
%
%   This is a popular example in the field of matrix iterations
%   of a matrix whose spectrum is in the right half-plane but
%   whose numerical range is not.  It's also a popular example
%   in the study of nonsymmetric Toeplitz matrices.  The matrix
%   was first described in [1] and its pseudospectra were first 
%   plotted in [2].
%
%   [1]: J. F. Grcar, "Operator coefficient methods for linear
%        equations", tech. report SAND89-8691, Sandia National
%        Labs, 1989
%   [2]: L. N. Trefethen, "Psuedospectra of matrices", in
%        "Numerical Analysis 1991" (Dundee 1991), Longman Sci.
%        Tech., Harlow, 1992, 234-266.

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  G = diag(ones(N,1),0) - diag(ones(N-1,1),-1) + ...
      diag(ones(N-1,1),1) + diag(ones(N-2,1),2) + ...
      diag(ones(N-3,1),3);
