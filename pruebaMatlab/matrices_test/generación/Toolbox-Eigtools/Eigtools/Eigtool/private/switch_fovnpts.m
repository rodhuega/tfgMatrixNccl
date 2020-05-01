function ps_data = switch_fovnpts(fig,cax,this_ver,ps_data)

% function ps_data = switch_fovnpts(fig,cax,this_ver,ps_data)
%
% Function called when the 'FoV npts' menu option is
% chosen.

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

    if isfield(ps_data,'fov_npts'),
      default_val = num2str(2*ps_data.fov_npts);
    else
      default_val = num2str(40);
    end;

    cont = 1;

    while cont==1,

      npts = inputdlg({['Please enter the number of points to use ' ...
                       'for the field of values computation (the larger this' ...
                       ' value, the better the accuracy, but the longer the ' ...
                       ' computation will take). This number must be even.']}, ...
                       'Field of Values Accuracy...', 1,{default_val});
      if isempty(npts) | isempty(npts{1}), % If cancel chosen (or blank), just do nothing
        return;
      else  % Take at least 10 of them
        thmax = max(10,str2num(npts{1}));
      end;

      if length(thmax)==1,
        if thmax>=0

% Ensure that the number of points is even
          if mod(thmax,2)==1, thmax = thmax-1; end;

% Translate the number of points to a number of angles
          thmax = thmax/2;

          ps_data.fov_npts = thmax;

% If fov is already displayed, pretend button pressed to ensure recomputed
% with new number of points
          mnu_itm_h = findobj(fig,'Tag','FieldOfVals');
          cur_state = get(mnu_itm_h,'value');
          if cur_state==1,
            ps_data = switch_fieldofvals(fig,cax,this_ver,ps_data);
          end;

          cont = 0;

        else
          h = errordlg('No. points must be non-negative','Invalid input','modal');
          waitfor(h);
        end;
      else
        h = errordlg('Invalid number for no. points','Invalid input','modal');
        waitfor(h);
      end;

    end;
