function T = tolosa_demo(N)

%   T = TOLOSA_DEMO(N) returns Matrix Market Tolosa
%   matrix (N can be 1090 or 4000). The routine mmread.m is 
%   supplied by the Matrix Market of NIST:
%    http://math.nist.gov/MatrixMarket/
%   For more information, please see
%    http://math.nist.gov/MatrixMarket/data/NEP/mvmtls/mvmtls.html

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  if N==1090,
    T = mmread('tols1090.mtx');
  elseif N==4000,
    T = mmread('tols4000.mtx');
  else
    error('Sorry, no data for that dimension.');
  end;
