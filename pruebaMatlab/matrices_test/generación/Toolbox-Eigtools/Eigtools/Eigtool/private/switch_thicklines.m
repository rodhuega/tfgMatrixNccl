function ps_data = switch_thicklines(fig,cax,this_ver,ps_data)

% function ps_data = switch_thicklines(fig,cax,this_ver,ps_data)
%
% Function called when the 'ThickLines' option is toggled

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

      mnu_itm_h = findobj(fig,'Tag','ThickLines');
      cur_state = get(mnu_itm_h,'checked');
      if strcmp(cur_state,'off'),
        set(mnu_itm_h,'checked','on');
        set(fig,'defaultpatchlinewidth',2);
        set(fig,'defaultlinelinewidth',2);
      else
        set(fig,'defaultpatchlinewidth',1);
        set(fig,'defaultlinelinewidth',1);
        set(mnu_itm_h,'checked','off');
      end;

      ps_data = switch_redrawcontour(fig,cax,this_ver,ps_data);
