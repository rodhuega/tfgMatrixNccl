function ps_data = switch_epslevmin(fig,cax,this_ver,ps_data)

% function ps_data = switch_epslevmin(fig,cax,this_ver,ps_data)
% 
% Function called when min eps level changed

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

    s = get(gcbo,'String');
    n = str2num(s);
    if ~isempty(n) & length(n)==1,
      if n<=ps_data.zoom_list{ps_data.zoom_pos}.levels.last,
        ps_data.zoom_list{ps_data.zoom_pos}.levels.first = n;
%% Now go to even levels
        ps_data.zoom_list{ps_data.zoom_pos}.levels.iseven = 1;
%% If there is already some contour data, replot it
        if isfield(ps_data.zoom_list{ps_data.zoom_pos},'x'),
          ps_data = switch_redrawcontour(fig,cax,this_ver,ps_data);
        end; 
      else
        errordlg('Minimum epsilon level must be smaller than maximum one!', ...
                 'Invalid input','modal');
      end;
    else
      errordlg('Invalid number for minimum epsilon level','Invalid input','modal');
    end;
