function toggle_mode_btn(fig,state,enable)

% function toggle_mode_btn(fig,state,enable)
%
% Function to toggle the state of the Go!/Stop! button
% in the figure pointed to by fig

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  h = findobj(fig,'Tag','EwCond');
  str = get(h,'String');

%% If state is set one way or the other, use this to determine
%% which way to change the button. If not, toggle it
  if nargin<2, state = ''; end;
  if strcmp(state,'Mode'),
    to_mode = 1;
  elseif strcmp(state,'RVec'),
    to_mode = 0;
  else
    if strcmp(str,'Mode+Cond.No'), % Toggle to Rvec
      to_mode = 0;
    else 
      to_mode = 1;
    end;
  end;


%% What state is it currently in?
  the_messages = get(findobj(fig,'Tag','MessageFrame'),'UserData');
  if to_mode==0, % Toggle to 'RVec'
    set(h,'Callback','eigtool_switch_fn(''RVecResid'');');
    set(h,'TooltipString',the_messages{34});
    set(h,'String','Ritz Vec+Resid');
    set(h,'BackgroundColor',[0.702 0.707 0.702]);  % mpe addition: keeps button rectangular
  else         % Toggle to 'Mode'
    set(h,'Callback','eigtool_switch_fn(''EwCond'');');
    set(h,'TooltipString',the_messages{12});
    set(h,'String','Mode+Cond.No');
    set(h,'BackgroundColor',[0.702 0.705 0.702]);  % mpe addition: keeps button rectangular
  end;

%% Enable the button whatever state it's now in if that was requested
  if nargin>=3 
    set(h,'Enable',enable);
  end;
