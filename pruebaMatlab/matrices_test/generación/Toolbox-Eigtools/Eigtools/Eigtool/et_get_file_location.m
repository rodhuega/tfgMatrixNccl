function loc = et_get_file_location(fname)

% function loc = et_get_file_location(fname)
%
% Function to find the full path to a particular MATLAB file,
% given the filename, fname.
%
% See also: WHICH

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

% Remove '.m' from filename, if there
if strcmp(fname(end-1:end),'.m'),
  fname = fname(1:end-2);
end;

% Get the full path, including the filename
floc = which(fname);

% Remove the filename to get just the path
% (remember to remove the '.m' from the filename too)
flen = length(fname);
loc = floc(1:end-flen-2);
