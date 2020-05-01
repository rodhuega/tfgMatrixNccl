function ps_data = switch_exportmtx(fig,cax,this_ver,ps_data)

% function ps_data = switch_exportmtx(fig,cax,this_ver,ps_data)
%
% Function called when the 'Export mtx' option is selected

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

      fn = inputdlg({['Please enter a name for the matrix variable in ' ...
                    'the base workspace.']}, ...
                    'Matrix variable name...', 1,{'A'});
      if isempty(fn) | isempty(fn{1}),        % If cancel chosen (or blank), just do nothing
        return;
      end;

      assignin('base',fn{1},ps_data.input_matrix);
