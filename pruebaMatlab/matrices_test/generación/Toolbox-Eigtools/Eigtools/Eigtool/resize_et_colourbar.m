function resize_et_colourbar(fig)

% function resize_et_colourbar(fig)
%
% Function to resize the colourbar to keep it the same
% width no matter how large the figure is.

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)


% Find the colourbar and set its units to pixels
cb_axes = findobj(fig,'Tag','MyColourBar');
cb_units = get(cb_axes,'units');
set(cb_axes,'units','pixels');

% Update the width
pcb = get(cb_axes,'pos');
pcb(3) = 36.35;

% Reset the position and return the units
set(cb_axes,'pos',pcb);
set(cb_axes,'units',cb_units);
