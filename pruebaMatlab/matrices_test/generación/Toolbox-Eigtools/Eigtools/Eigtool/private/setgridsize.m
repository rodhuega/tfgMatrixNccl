function npts = setgridsize(n,the_min,the_max,iscomplex)

% function npts = setgridsize(n)
%
% Determine a grid size which will result in a quick computation
%
% n       is the size of the matrix whose pseudospectra we're computing
% the_min is the minimum grid size to use
% the_max is the maximum grid size to use
%
% npts    is the size of the grid to use for this matrix

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

%% This relation determined by experiment...
%% 300/x^(5/6) gives approx the number of points to use. Then use max 100, min 15.
%% Default to a fast grid whose size is determined by this relation.
npts = round(min(max(300/n^(5/6), the_min), the_max));

%% Reduce the size if the matrix is complex
if iscomplex==1,
  npts = max(the_min,floor(3*npts/4));
end;
if mod(npts,2)==1, npts = npts+1; end;
