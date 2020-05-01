function A = transient_demo(N)

%   A = TRANSIENT_DEMO(N) returns matrix A of dimension N.
%
%   The matrix in this demo is designed to have transient
%   behaviour in both ||A^k|| and ||e^{tA}||. To see this,
%   run the demo, then go to the `Transients' menu to view
%   the behaviour. Lower bounds on the transient behaviour
%   can be plotted by selecting items from the same menu.

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  x = 2*pi*(0:N-1)/N;
  D = diag(ones(N-1,1),1); D(N,1) = 1;
  A = diag(exp(1i*x)) + D;
  A = .4*A - .5*eye(N);
