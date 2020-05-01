function delete_et_marker(pmfig,guifig,no_fig_delete)

% function delete_et_marker(pmfig,guifig,no_fig_delete)
%
% When a pseudomode is plotted, a marker is drawn in the GUI
% window indicating the corresponding pseudoeigenvalue. This
% function is set to be the CloseRequestFcn of the pseudomode
% window so that the marker can be deleted when this window
% is closed. It is not intended to be called manually by a 
% user.
%
% pmfig     the handle to the pseudmode figure
% guifig    the handle to the parent GUI
% no_fig_delete   set to 1 to keep the figure

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)
 
% If the GUI is still open
if ishandle(guifig),

% Get the GUI's data to get the handle of the circle to delete
  ps_data = get(guifig,'userdata');
  if length(ps_data.mode_markers)>=pmfig & ~isempty(ps_data.mode_markers{pmfig}) ...
     & ishandle(ps_data.mode_markers{pmfig}.h),
    delete(ps_data.mode_markers{pmfig}.h);

% Remove this entry from the GUI's list
    ps_data.mode_markers{pmfig} = [];

% Save the update to the GUI data
    set(guifig,'userdata',ps_data);
  end;
end;

% Close the pseudmode figure
if nargin<3 | no_fig_delete==0, delete(pmfig); end;
