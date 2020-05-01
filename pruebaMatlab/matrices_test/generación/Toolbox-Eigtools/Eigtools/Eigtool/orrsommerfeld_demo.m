function C = orrsommerfeld_demo(N)

%   C = ORRSOMMERFELD_DEMO(N) returns matrix C of
%   dimension N.
%
%   D is an Orr-Sommerfeld operator generated from a spectral
%   method [2]. This discretization results in a matrix of
%   dimension N=99. There are numerous spurious eigenvalues
%   beyond the plotted axis, so the matrix is projected
%   onto an appropriate invariant subspace to accelerate
%   the computation. Pseudospectra of Orr-Sommerfeld
%   operators were first investigated in [1].
%
%   [1]: S. C. Reddy, P. J. Schmidt and D. S. Henningson,
%        "Pseudospectra of the Orr-Sommerfeld operator",
%        SIAM J. Appl. Math. 53(1), pp. 15-47, 1993.
%   [2]: L. N. Trefethen, "Spectral methods in MATLAB",
%        SIAM, 2000.

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  R = 5772;
  [D,x] = cheb(N);     %   cheb.m from Trefethen's "Spectral Methods in MATLAB"
  D2 = D^2; D2 = D2(2:N,2:N);
  S = diag([0; 1 ./(1-x(2:N).^2); 0]);
  D4 = (diag(1-x.^2)*D^4 - 8*diag(x)*D^3 - 12*D^2)*S;
  D4 = D4(2:N,2:N);
  I = eye(N-1);
  A = (D4-2*D2+I)/R - 2i*I - 1i*diag(1-x(2:N).^2)*(D2-I);
  B = D2-I;
  C = B\A;
