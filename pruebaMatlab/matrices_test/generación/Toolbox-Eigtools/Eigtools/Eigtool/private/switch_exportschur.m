function ps_data = switch_exportschur(fig,cax,this_ver,ps_data)

% function ps_data = switch_exportschur(fig,cax,this_ver,ps_data)
%
% Function called when the 'Export Schur factor' option is selected

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

      fn = inputdlg({['Please enter a name for the Schur factor variable in ' ...
                    'the base workspace.']}, ...
                    'Schur factor variable name...', 1,{'T'});
      if isempty(fn) | isempty(fn{1}),        % If cancel chosen (or blank), just do nothing
        return;
      end;

      assignin('base',fn{1},ps_data.schur_matrix);
