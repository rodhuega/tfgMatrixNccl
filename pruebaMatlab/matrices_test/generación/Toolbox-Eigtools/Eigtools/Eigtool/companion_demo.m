function E = companion_demo(N)

%   E = COMPANION_DEMO(N) returns matrix E of
%   dimension N.
%
%   This matrix was suggested by Cleve Moler. It is the
%   companion matrix for the truncated power series of
%   exp(z). Note that serious rounding errors for N much
%   larger than 15 due to the large size of some entries.

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  c = [1 1 ./ cumprod(1:N)];
  E = compan(fliplr(c));
