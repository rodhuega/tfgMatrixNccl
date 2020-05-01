function C = convdiff_demo(N)

%   C = CONVDIFF_DEMO(N) returns matrix C of dimension N.
%
%   C is the 1-D convection-diffusion operator eu'' + u' on [0,1] with
%   u(0)=u(1)=0 for e=1/30, discretised using a Chebyshev spectral
%   method [1]. The boundaries of pseudospectra of the infinite
%   dimensional operator are shaped approximately like parabolas,
%   as already revealed on the right part of this low-dimensional
%   approximation [2].
%
%   [1]: L. N. Trefethen, "Spectral methods in MATLAB",
%        SIAM, 2000.
%   [2]: S. C. Reddy and L. N. Trefethen, "Pseudospectra of the
%        convection-diffusion operator", SIAM J. Appl. Math. 54(6),
%        pp. 1634-1649, 1994.

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  e = 1/30;
  [D,x] = cheb(N); %   cheb.m from Trefethen's "Spectral Methods in MATLAB"
  C = e*D^2 + D;
  C = C(2:end-1,2:end-1); 
