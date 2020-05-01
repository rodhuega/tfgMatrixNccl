function shrink_gui(fig,scale)

% function shrink_gui(fig,scale)
%
% Function to reduce the size of the GUI and all its controls
% for example if using a laptop screen

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

cs = get(fig,'children');

% Scale all the objects in the figure
for i=1:length(cs),

  p = get(cs(i),'position');

  if isempty(findstr('Menu',get(cs(i),'Tag'))),  set(cs(i),'position',scale*p); end;

end;

% Scale the figure itself
p = get(fig,'position');
set(fig,'position',scale*p);

% Reduce the fontsize in the message text a bit
h = findobj(fig,'Tag','MessageText');
set(h,'fontsize',12);
