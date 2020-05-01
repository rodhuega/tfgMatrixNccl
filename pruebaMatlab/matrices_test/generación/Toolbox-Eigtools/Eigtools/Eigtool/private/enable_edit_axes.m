function enable_edit_axes(fig)

% function enable_edit_axes(fig)
%
% Function to turn on the edit text boxes for the axes

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  the_handle = findobj(fig,'Tag','MainAxes');
  set(the_handle,'HitTest','on');
  set(fig,'WindowButtonDownFcn','eigtool_switch_fn(''PsArea'');');

  the_handle = findobj(fig,'Tag','xmin');
  set(the_handle,'Enable','on');
  the_handle = findobj(fig,'Tag','xmax');
  set(the_handle,'Enable','on');
  the_handle = findobj(fig,'Tag','ymin');
  set(the_handle,'Enable','on');
  the_handle = findobj(fig,'Tag','ymax');
  set(the_handle,'Enable','on');
  the_handle = findobj(fig,'Tag','ScaleEqual');
  set(the_handle,'Enable','on');

