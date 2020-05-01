function M = rdbrusselator_demo(N)

%   M = RDBRUSSELATOR_DEMO(N) returns Matrix Market reaction-diffusion
%   brusselator model (N can be 800 or 3200). The routine mmread.m is 
%   supplied by the Matrix Market of NIST:
%    http://math.nist.gov/MatrixMarket/
%   For more information, please see
%    http://math.nist.gov/MatrixMarket/data/NEP/brussel/brussel.html

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  if N==800,
    M = mmread('rdb800l.mtx');
  elseif N==3200,
    M = mmread('rdb3200l.mtx');
  else
    error('Sorry, no data for that dimension.');
  end;
