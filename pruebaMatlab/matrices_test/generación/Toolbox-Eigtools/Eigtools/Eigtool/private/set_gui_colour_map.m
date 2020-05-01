function cm = set_gui_colour_map(fig)

% function cm = set_gui_colour_map(fig)
%
% Function to set the colourmap of a figure to the standard
% one used by the GUI.

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

cm = getpref('EigTool','colormap');
set(fig,'colormap',cm);
