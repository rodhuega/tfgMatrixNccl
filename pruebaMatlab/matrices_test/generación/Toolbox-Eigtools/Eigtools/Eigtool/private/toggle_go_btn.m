function ps_data = toggle_go_btn(fig,state,enable,ps_data)

% function toggle_go_btn(fig,state,enable)
%
% Function to toggle the state of the Go!/Stop! button
% in the figure pointed to by fig

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)
  
  h = findobj(fig,'Tag','RedrawPlot');
  str = get(h,'String');

%% If state is set one way or the other, use this to determine
%% which way to change the button. If not, toggle it
  if nargin<2, state = ''; end;
  if strcmp(state,'Go!'),
    to_go = 1;
  elseif strcmp(state,'Stop!'),
    to_go = 0;
  else
    if strcmp(str,'Go!'), % Toggle to 'Stop!'
      to_go = 0;
    else 
      to_go = 1;
    end;
  end;


%% What state is it currently in?
  the_messages = get(findobj(fig,'Tag','MessageFrame'),'UserData');
  if to_go==0, % Toggle to 'Stop!'
    set(h,'BackgroundColor',[1 0.352291792379578 0.25431683106714]);
    set(h,'Callback','eigtool_switch_fn(''STOP'');');
    set(h,'TooltipString',the_messages{20});
    set(h,'String','Stop!');
  else         % Toggle to 'Go!'
    set(h,'BackgroundColor',[0 0.921 0.38375]);
    set(h,'Callback','eigtool_switch_fn(''Redraw'');');
    set(h,'TooltipString',the_messages{9});
    set(h,'String','Go!');
  end;

%% Enable the button whatever state it's now in if that was requested
%% (and remember it for later)
  if nargin>=3 
    ps_data.go_btn_state = enable;
    set(h,'Enable',enable);
  end;
