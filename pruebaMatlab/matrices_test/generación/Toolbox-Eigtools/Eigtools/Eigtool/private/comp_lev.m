function lev_desc = comp_lev(levels)

% function lev_desc = comp_lev(levels)
%
% This function compresses the full vector form of the levels
% to display to a three-position compressed form containing
% the start level, the step size and the end level.
%
% levels      The levels to compress
%
% lev_desc    The compressed description

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

% Find out if they're not evenly spaced
if max(abs(diff(diff(levels))))>3*eps,
  lev_desc.iseven = 0;
  lev_desc.full_levels = levels;
else
  lev_desc.iseven = 1;
end;

% Assign the standard descriptor
lev_desc.first = levels(1);
if length(levels)>1,
  lev_desc.step = min(diff(levels));
else
  lev_desc.step = 0;
end;
lev_desc.last = levels(end);
