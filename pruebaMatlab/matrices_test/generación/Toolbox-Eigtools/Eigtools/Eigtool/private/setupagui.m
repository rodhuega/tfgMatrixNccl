% Script called after aeigs has finished to store the final
% variables so that the GUI can get at them.

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

% Get the matrices H and V (called thev)
  extract_mtx;

% Append them to the existing data
  ps_data = get(opts.eigtool,'userdata');
  ps_data.proj_matrix = H;
  ps_data.proj_unitary_mtx = thev;
  set(opts.eigtool,'userdata',ps_data);
