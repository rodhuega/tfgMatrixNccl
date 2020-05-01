function step = get_step_size(n,ly,routine)

% function step = get_step_size(n,ly)
% Function to compute a stepsize which will allow as many rows
% of the grid to be done as possible within the psacore***
% mexfiles, but will still allow the waitbar to be updated
% sufficiently often.
%
% n        The dimension of the matrix
% ly       The number of gridpoints in the y direction
% routine  The name of the routine which will use the step size
%
% step     The number of rows of the grid to compute at once

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

% Generate a stepsize
if n<100, step = max(1,floor(ly/8));
else step = min(ly,max(1,floor(4*ly/n))); end;

% If we can see the m-file (i.e. no mex file),
% decrease the stepsize as the computation will take
% much longer
if exist(routine,'file')==2,
  step = max(1,floor(step/4));
end;
