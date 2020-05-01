function c = findcontour(cdat)

% function c = findcontour(cdat)
%
% Extract the actual contour levels plotted on a contour plot
%
% cdat        the contour data as output by contour
%
% c           a vector of the levels plotted

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)
% Based on a routine by Mark Embree

%% Initialise
c = [];
j = 1;

%% Loop over the data
while j<size(cdat,2)
  c = [c;cdat(1,j)];
  j = j+cdat(2,j)+1;
end

%% Tidy up the output
c = sort(unique(c));
