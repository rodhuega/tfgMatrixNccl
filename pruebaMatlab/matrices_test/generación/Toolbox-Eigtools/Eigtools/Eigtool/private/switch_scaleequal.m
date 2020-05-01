function ps_data = switch_scaleequal(fig,cax,this_ver,ps_data)

% function ps_data = switch_scaleequal(fig,cax,this_ver,ps_data)
%
% Function called when scale-equal is toggled

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

%% Store the current values so that they can be set after
%% changing the axis - don't want axis to grow if set to 
%% equal, want to constrain it.
    set(fig,'CurrentAxes',cax);

    if this_ver<6,
      cur_axis = axis;
    else
      cur_axis = axis(cax);
    end;

    if get(findobj(fig,'Tag','ScaleEqual'),'Value')==1, %% Currently on equal scale
      if this_ver<6,
        axis equal;
      else
        axis(cax,'equal');
      end;
    else                     %% not on equal scale
      if this_ver<6,
        axis normal;
      else
        axis(cax,'normal');
      end;
    end;
    if this_ver<6,
      axis(cur_axis);
    else
      axis(cax,cur_axis);
    end;

%% Make sure the main axes are active, so that the 
%% axis call in the parameter list to set_edit_text
%% is valid
    if this_ver<6,
      set(fig,'CurrentAxes',cax);
      the_ax = axis;
    else
      the_ax = axis(cax);
    end;
    set_edit_text(fig,the_ax,ps_data.zoom_list{ps_data.zoom_pos}.levels, ...
                  ps_data.zoom_list{ps_data.zoom_pos}.npts, ...
                  ps_data.zoom_list{ps_data.zoom_pos}.proj_lev);

%% Enable the 'Go!' button now we've changed the axis
    ps_data = toggle_go_btn(fig,'Go!','on',ps_data);
    ps_data = update_messagebar(fig,ps_data,28);
