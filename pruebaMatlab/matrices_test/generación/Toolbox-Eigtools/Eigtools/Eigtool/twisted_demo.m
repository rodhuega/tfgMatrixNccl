function A = twisted_demo(N)

%   A = TWISTED_DEMO(N) returns matrix A of dimension N.
%
%   The "cross" matrix is an example of a banded "twisted circulant matrix".
%   It is tridiagonal (with periodic wraparound) but instead of having
%   constant diagonals, as in a true circulant matrix, it has a smoothly
%   varying main diagonal.  The result is an exponentially strong degree
%   of nonnormality, with pseudomodes in the form of wave packets.
%   See [1].
%   
%   [1]: J. Chapman and L. N. Trefethen, "Wave packet pseudomodes of twisted
%        Toeplitz matrices", to appear.

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  x = 2*pi*(0:N-1)/N;
  D = diag(ones(N-1,1),1); D(N,1) = 1;
  A = diag(2*sin(x)) + D - D';
