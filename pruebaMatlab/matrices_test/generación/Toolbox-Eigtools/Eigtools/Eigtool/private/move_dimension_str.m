function move_dimension_str(fig,the_ax)

% function move_dimension_str(fig,the_ax)
%
% Function to move the string showing the dimension of the
% matrix when the axis limits are changed

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

%% Move the dimension, if it is on:
  mnu_itm_h = findobj(fig,'Tag','ShowDimension');
  cur_state = get(mnu_itm_h,'checked');
  if strcmp(cur_state,'on'),
%% Move the text
    hdl = findobj(fig,'Tag','DimText');
    set(hdl,'units','pixels');
    set(hdl,'position',[15 15 0]);
  end;
