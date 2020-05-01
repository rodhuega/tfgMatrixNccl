function M = olmstead_demo(N)

%   M = OLMSTEAD_DEMO(N) returns Matrix Market Olmstead
%   matrix (N can be 500 or 1000). The routine mmread.m is 
%   supplied by the Matrix Market of NIST:
%    http://math.nist.gov/MatrixMarket/
%   For more information, please see
%    http://math.nist.gov/MatrixMarket/data/NEP/olmstead/olmstead.html

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  if N==500,
    M = mmread('olm500.mtx');
  elseif N==1000,
    M = mmread('olm1000.mtx');
  else
    error('Sorry, no data for that dimension.');
  end;
