function ps_data = update_messagebar(fig,ps_data,new_message,no_drawnow)

% function prev_message = update_messagebar(new_message)
%
% Function to change the message on display in the message bar. 
% It returns the current message so that it can be returned
% to at a later date if necessary.
%
% fig         Handle to the current EIGTOOL figure
% ps_data     The data for the current EIGTOOL
% new_message A message number to switch to, or 
%                'prev' to use the previous one
% no_drawnow  (optional) set to 1 to prevent drawnow
% 
% ps_data     The changed EIGTOOL data

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

% Use the previous message if requested
    if strcmp(new_message,'prev'),
      new_message = ps_data.last_message;
    end;

% Return the current message so it can be returned to later
    ps_data.last_message = ps_data.current_message;

    text_handle = findobj(fig,'Tag','MessageText');
    msgs = get(findobj(fig,'Tag','MessageFrame'),'UserData');

% Set the new message
    ps_data.current_message = new_message;
    set(text_handle,'String', msgs{ps_data.current_message});

% Make sure it's displayed, if requested
    if nargin<4 | no_drawnow==0,
      drawnow;
    end;
