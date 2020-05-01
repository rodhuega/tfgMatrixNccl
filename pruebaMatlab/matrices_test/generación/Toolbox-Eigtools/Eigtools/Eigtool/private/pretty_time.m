function [total_time,the_units] = pretty_time(total_time)

% function [total_time,the_units] = pretty_time(total_time)
%
% Function to convert time in seconds to a whole number of
% hours, minutes or seconds depending on the size.

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  if total_time<180,
    total_time = ceil(total_time);
    the_units = 'seconds';
  elseif total_time<10800,
    total_time = ceil(total_time/60);
    the_units = 'minutes';
  else % Display in hours
    total_time = round(total_time/360)/10;
    the_units = 'hours';
  end;
