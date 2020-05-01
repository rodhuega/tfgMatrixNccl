function ps_data = switch_progrvals(fig,cax,this_ver,ps_data)

% function ps_data = switch_progrvals(fig,cax,this_ver,ps_data)
%
% Function called when the 'Show prog using Ritz vals' menu option is
% chosen.

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

      mnu_itm_h = findobj(fig,'Tag','ProgRVals');
      cur_state = get(mnu_itm_h,'checked');
      if strcmp(cur_state,'off'),
        set(mnu_itm_h,'checked','on');
      else
        set(mnu_itm_h,'checked','off');
      end;
