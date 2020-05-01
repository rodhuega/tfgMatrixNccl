function ps_data = switch_display_colourbar(fig,cax,this_ver,ps_data)

% function ps_data = switch_display_colourbar(fig,cax,this_ver,ps_data)
%
% Function called when the 'Display Colourbar' menu option is
% chosen.

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

      mnu_itm_h = findobj(fig,'Tag','DisplayColourbar');
      cur_state = get(mnu_itm_h,'checked');
      if strcmp(cur_state,'off'),
        set(mnu_itm_h,'checked','on');
        grow_main_axes(fig,0);
      else
        set(mnu_itm_h,'checked','off');
        grow_main_axes(fig,1);
      end;
