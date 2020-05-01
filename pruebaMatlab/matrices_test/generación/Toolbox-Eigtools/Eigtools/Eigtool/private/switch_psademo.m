function switch_psademo

% function switch_safety
%
% Function called when Pseudospectra Tutorial menu option is chosen

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  h = msgbox(['To learn more about pseudospectra and their applications, ' ...
              'type   psademo   at the MATLAB command prompt; you will be ' ...
              'led through a step-by-step guide to pseudospectra and '  ...
              'their applications.'],'modal');
  waitfor(h);

