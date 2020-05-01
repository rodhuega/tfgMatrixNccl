function A = airy_demo(N)

%   A = AIRY_DEMO(N) returns matrix A of dimension N.
%
%   Our "Airy operator" is an operator acting on functions of x
%   defined on [-1,1] according to the formula
%   
%              L u = epsilon*(d^2 u / dx^2) + i*x*u
%   
%   where epsilon is a small parameter.  The spectrum of this operator
%   is an unbounded discrete set contained in the half-strip Re z < 0,
%   -1 < Im z < 1.  The pseudospectra approximately fill the whole 
%   strip.  The pseudospectra of this operator were first considered
%   by Reddy, Schmid and Henningson as a model of the more complicated
%   Orr-Sommerfeld problem [1].  Our M-file is based on a Chebyshev
%   collocation spectral discretization with epsilon = 4e-3.
%
%   [1]: S. C. Reddy, P. J. Schmidt and D. S. Henningson,
%        "Pseudospectra of the Orr-Sommerfeld operator",
%        SIAM J. Appl. Math. 53(1), pp. 15-47, 1993.

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  [D,x] = cheb(N);
  D2 = D^2;
  D = D(2:N,2:N); D2 = D2(2:N,2:N); x = x(2:N);
  A = 3e-4*D2 + 1i*diag(x);
