function ps_data = switch_meshsize(fig,cax,ps_data)

% function ps_data = switch_meshsize(fig,cax,ps_data)
%
% Function called when the mesh size is changed

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

    s = get(gcbo,'String');
    n = str2num(s);
    if ~isempty(n) & length(n)==1,
      if n>4
        ps_data.zoom_list{ps_data.zoom_pos}.npts=n;
        ps_data = update_messagebar(fig,ps_data,3);

%% Enable the 'Go!' button now we've changed the mesh size
        ps_data = toggle_go_btn(fig,'Go!','on',ps_data);

      else
        errordlg('Grid must be at least 5.','Invalid input','modal');
      end;
    else
      errordlg('Invalid number for grid size','Invalid input','modal');
    end;
