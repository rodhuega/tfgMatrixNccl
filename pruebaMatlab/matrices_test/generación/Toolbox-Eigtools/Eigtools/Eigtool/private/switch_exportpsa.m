function ps_data = switch_exportpsa(fig,cax,this_ver,ps_data)

% function ps_data = switch_exportpsa(fig,cax,this_ver,ps_data)
%
% Function called when the 'Export psa' option is selected

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

% Ask for the variable names
   prompt={'Variable name for x grid points:', ...
           'Variable name for y grid points:', ...
           'Variable name for singular value data:'};
   def={'x','y','Z'};
   dlgTitle='Export pseudospectra data...';
   lineNo=1;
   fn=inputdlg(prompt,dlgTitle,lineNo,def);

   if isempty(fn),      % If cancel chosen, just do nothing
     return;
   end;

% Save the data to the base workspace if the names not blank
   if ~isempty(fn{1}),
     assignin('base',fn{1},ps_data.zoom_list{ps_data.zoom_pos}.x);
   end;
   if ~isempty(fn{2}),
     assignin('base',fn{2},ps_data.zoom_list{ps_data.zoom_pos}.y);
   end;
   if ~isempty(fn{3}),
     assignin('base',fn{3},ps_data.zoom_list{ps_data.zoom_pos}.Z);
   end;
