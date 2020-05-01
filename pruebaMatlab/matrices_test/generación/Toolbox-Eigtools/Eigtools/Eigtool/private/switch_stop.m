function switch_stop(fig)

% function switch_stop(fig)
%
% Function called when the 'Stop!' button is pressed

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

%% This global variable is checked by the PS computation routines
    global stop_comp;

%% Set this to be the handle of the current figure; that way if 
%% several instances of the GUI are going at once, only this one
%% will be stopped
    stop_comp = fig;
