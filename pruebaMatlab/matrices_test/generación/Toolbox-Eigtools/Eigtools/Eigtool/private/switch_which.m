function ps_data = switch_which(fig,cax,this_ver,ps_data,set_only)

% function ps_data = switch_which(fig,cax,this_ver,ps_data)
%
% Function called when the 'Which' menu is
% clicked

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

% Which ones are they?
    which_hdl = findobj(fig,'Tag','Which');
    which_vals = get(which_hdl,'userdata');
    which = which_vals{get(which_hdl,'Value')};

% This is basically shift and invert, which we don't currently support
    if strcmp(which,'SM'), 
      h = errordlg('Cannot currently do SM eigenvalues (= shift & invert).','Error...','modal');
      waitfor(h);
% Revert back
      prev = find(strcmp(which_vals,ps_data.ARPACK_which));
      set(findobj(fig,'Tag','Which'),'Value',prev);
      return;
     end;

% If this is not the current one, change it
    if nargin<5 | set_only==0,
      if ~strcmp(which,ps_data.ARPACK_which),
        ps_data.ARPACK_which = which;
        ps_data.proj_valid = 0;
        ps_data = toggle_go_btn(fig,'Go!','on',ps_data);
        ps_data = update_messagebar(fig,ps_data,31,1);
      end;
    else
      ps_data.ARPACK_which = which;
    end;
  
  
