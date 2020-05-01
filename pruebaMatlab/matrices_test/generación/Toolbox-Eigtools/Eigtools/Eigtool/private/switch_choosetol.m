function ps_data = switch_choosetol(fig,cax,this_ver,ps_data)

% function ps_data = switch_choosetol(fig,cax,this_ver,ps_data)
%
% Function called when the 'Choose Tol' menu option is
% chosen.

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

    if isfield(ps_data,'ARPACK_tol') & ~strcmp(ps_data.ARPACK_tol,'auto'),
      default_val = num2str(ps_data.ARPACK_tol);
    else
      default_val = '';
    end;

    cont = 1;

    while cont==1,

      s = inputdlg({['Please enter the tolerance to use within ARPACK ' ...
                   ' (leave blank for the default of machine epsilon).']}, ...
                   'ARPACK tolerance...', 1,{default_val});
      if isempty(s),        % If cancel chosen, just do nothing
        return;
      elseif isempty(s{1}), % If left blank, remove the field to use default val.
        if isfield(ps_data,'ARPACK_tol'),
          ps_data = rmfield(ps_data,'ARPACK_tol');
        end;
        return;
      else
        tol = str2num(s{1});
      end;

      if length(tol)==1,
        if tol>=0

          ps_data.ARPACK_tol = tol;
          ps_data = update_messagebar(fig,ps_data,31);

%% Enable the 'Go!' button now we've changed the parameters
          ps_data = toggle_go_btn(fig,'Go!','on',ps_data);

%% The current projection is no longer valid
          ps_data.proj_valid = 0;

          cont = 0;

        else
          h = errordlg('Tolerance must be non-negative','Invalid input','modal');
          waitfor(h);
        end;
      else
        h = errordlg('Invalid number for tolerance','Invalid input','modal');
        waitfor(h);
      end;

    end;
