function ps_data = switch_showgrid(fig,cax,this_ver,ps_data)

% function ps_data = switch_showgrid(fig,cax,this_ver,ps_data)
%
% Function called when the 'Show Grid' menu option is
% chosen.

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

      mnu_itm_h = findobj(fig,'Tag','ShowGrid');
      cur_state = get(mnu_itm_h,'checked');
      if strcmp(cur_state,'off'),
        set(mnu_itm_h,'checked','on');
        ps_data = switch_redrawcontour(fig,cax,this_ver,ps_data);
      else
%% The code for 'RedrawContour' should store the handle for the grid
%% in the userdata field for the menu item
        delete(findobj(fig,'Tag','GridText'));
        set(mnu_itm_h,'checked','off');
      end;
