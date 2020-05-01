function ps_data = switch_choosev0(fig,cax,this_ver,ps_data)

% function ps_data = switch_choosev0(fig,cax,this_ver,ps_data)
%
% Function called when the 'Choose v0' menu option is
% chosen.

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

    if isfield(ps_data,'ARPACK_v0_cmd') ...
      & isfield(ps_data,'ARPACK_v0') & ~strcmp(ps_data.ARPACK_v0,'auto'),
      default_val = ps_data.ARPACK_v0_cmd;
    else
      default_val = '';
    end;

    cont = 1;

    while cont==1,

      s = inputdlg({['Please enter a command to create the starting vector ' ...
                   'v0 (leave blank for default random vector).']}, ...
                   'Starting vector...', 1,{default_val});
      if isempty(s),        % If cancel chosen, just do nothing
        return;
      elseif isempty(s{1}), % If left blank, remove the field to use default val.
        if isfield(ps_data,'ARPACK_v0'),
          ps_data = rmfield(ps_data,'ARPACK_v0');
        end;
        if isfield(ps_data,'ARPACK_v0_cmd'),
          ps_data = rmfield(ps_data,'ARPACK_v0_cmd');
        end;
        return;
      else

%% Remove any trailing blanks
          the_cmd = deblank(char(s{1}));

%% If v0 set via opts.v0, and nothing changed here, just leave.
          if strcmp(the_cmd,'<Set using input options>'),
            return;
          end;

%% Remove ; if it is at the end - causes eval to fail below
          if the_cmd(end)==';',
            the_cmd = the_cmd(1:end-1);
          end;

%% Execute the command to get the data - set A to [] if fail...
          v0 = evalin('base',the_cmd,'''error''');

          if strcmp('error',v0),
            the_err = get(0,'ErrorMessage');
            h=errordlg(['Error: ', ...
                        the_err ' Please try again.'],'modal'); 
            waitfor(h);
            default_val = s{1};
          else

            ps_data.ARPACK_v0 = v0;
% Store the command too for when we return to this dialog
            ps_data.ARPACK_v0_cmd = the_cmd;
            ps_data = update_messagebar(fig,ps_data,31);

%% Enable the 'Go!' button now we've changed the parameters
            ps_data = toggle_go_btn(fig,'Go!','on',ps_data);

%% The current projection is no longer valid
            ps_data.proj_valid = 0;

            cont = 0;

          end;
     end;

    end;
