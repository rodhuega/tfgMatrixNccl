function M = pde_demo(N)

%   M = PDE_DEMO(N) returns Matrix Market PDE
%   matrix (N can be 900 or 2961). The routine mmread.m is 
%   supplied by the Matrix Market of NIST:
%    http://math.nist.gov/MatrixMarket/
%   For more information, please see
%    http://math.nist.gov/MatrixMarket/data/NEP/matpde/matpde.html

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  if N==900,
    M = mmread('pde900.mtx');
  elseif N==2961,
    M = mmread('pde2961.mtx');
  else
    error('Sorry, no data for that dimension.');
  end;
