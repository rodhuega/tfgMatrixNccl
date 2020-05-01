function M = dwave_demo(N)

%   M = DWAVE_DEMO(N) returns Matrix Market dielectric channel
%   waveguide problem matrix (N can only be 2048). The routine 
%   mmread.m is supplied by the Matrix Market of NIST:
%    http://math.nist.gov/MatrixMarket/
%   For more information, please see
%    http://math.nist.gov/MatrixMarket/data/NEP/dwave/dwave.html

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  if N==2048,
    M = mmread('dw2048.mtx');
  else
    error('Sorry, no data for that dimension.');
  end;
