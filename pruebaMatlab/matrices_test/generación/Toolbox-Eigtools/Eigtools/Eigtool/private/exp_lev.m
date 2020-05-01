function levels = exp_lev(lev_desc)

% function levels = exp_lev(lev_desc)
%
% This function expands the compressed vector form of the levels
% to full vector form.
%
% lev_desc    The compressed description
%
% levels      The expanded levels

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

% If the levels are evenly spaced
if lev_desc.iseven==1,
  if lev_desc.step~=0,
    levels = fliplr([lev_desc.last:-lev_desc.step:lev_desc.first]);
  else
    levels = [lev_desc.first lev_desc.first];
  end;
else
  levels = lev_desc.full_levels;
  if length(levels)==1,
    levels = levels*[1 1];
  end;
end;
