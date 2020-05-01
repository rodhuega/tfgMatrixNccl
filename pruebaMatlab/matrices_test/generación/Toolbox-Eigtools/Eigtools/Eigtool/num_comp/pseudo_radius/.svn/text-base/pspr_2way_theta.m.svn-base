function thetanew = pspr_2way_theta(A, mE, epsln, r, thetawant, iter, radtol, ...
     smalltol, plotfig, prtlevel,h)
% Michael Overton and Emre Mengi
% called by pspr_2way.m
% Given a radius r, it first computes the intersection points
% of eps-pseudospectrum boundary with the circle with radius r.
% This is achieved by finding the generalized eigenvalues of the 
% matrix pencil F - lambda*G where
%
%       F=[-eps*I A;r*I 0], G=[0 r*I; A' -eps*I]
%
% and then performing a singular value test. The singular value test
% is neccessary to eliminate the points for which eps is a singular
% value but not the smallest one. Finally the midpoints of two
% succesive intersection points on the circle with radius r is
% calculated, i.e. let re^(i*theta_i) and re^(i*theta_(i+1)) be the 
% ith and (i+1)th intersection points, then ith midpoint is 
% re^(i*(theta_i+theta_(i+1))/2). Special attention is paid to
% keep the angles in the interval [-pi,pi). Furthermore, as it
% was the case for pseudo-absicca code, we specifically considered
% the case when the angle from the previous iteration is contained
% in one of the intervals. At the exit thetanew contains the 
% angles of the midpoints.


n = length(A);

% compute the generalized eigenvalues of the matrix pencil F - lambda*G
R = r * eye(n);
O = 0 * eye(n);
F = [mE A; R O];
G = [O R; A' mE];
eM = eig(F,G);

% extract the eigenvalues with magnitude 1
% a small tolerance is used
ind = find((abs(eM) < (1 + radtol)) & (abs(eM) > (1 - radtol)));
eM = eM(ind);

       
if (isempty(eM)) % check if M has an eigenvalue with magnitude 1
   thetanew = []; 
else

   % sort eM wrt theta values
   [theta, indx] = sort(angle(eM));
   theta = angle(eM(indx));
          
         
   
  
   % perform singular value test on the points probably on
   % eps-pseudospectrum boundary.
   % the ones actually on the eps-pseudospectrum are those with smallest 
   % singular value equal to eps
   indx2 = [];
   for j = 1: length(theta)
       Ashift = A - (r*(cos(theta(j)) + i*sin(theta(j))))*eye(n);
       s = svd(Ashift);
       [minval,minind] = min(abs(s-epsln));
       
       if minind == n
           indx2 = [indx2; j];   % accept this eigenvalue
       end
   end
      
   removed = length(theta) - length(indx2);
   
   if removed > 0
       if prtlevel > 0
           fprintf('\npspr_2way_theta: singular value test removed %d eigenvalues ', removed)
       end
       theta = theta(indx2);
   end
   
   
   if (isempty(theta))
        if (n >= 30)
            close(h)
        end
       error('singular value test removed all of the intersection points(please try smaller epsilon)')
   end
   
       
   
   % organize in pairs and take midpoints
   ind = 0;

   % shift thetawant, the angle from the previous iteration, into 
   % the interval [0,2pi]
   if (thetawant < 0)
       thetawant = thetawant + 2*pi;
   end

   
   for j=1:length(theta)     
      thetalow = theta(j);

      
      % shift thetalow into the interval [0,2pi]
      if (thetalow < 0)
          thetalow = thetalow + 2*pi;
      end
      
      
      
      if (j < length(theta))
          thetahigh = theta(j+1);
      else
          thetahigh = theta(1);
      end

      
      
      % shift thetahigh into the interval [0,2pi]
      if (thetahigh < 0)
              thetahigh = thetahigh + 2*pi;
      end
      
      
      % if thetahigh is smaller than thetalow, shift thetahigh
      % furthermore
      if (thetahigh <= thetalow)
          thetahigh = thetahigh + 2*pi;
      end
      
      
          
      % before taking the midpoint, if this interval is not very short,
      % check and see if thetawant is in this interval, well away from the
      % end points.  If so, break this pair into two pairs, one above
      % and one below thetawant, and take midpoints of both.
      inttol = .01 * (thetahigh - thetalow);
      if thetawant > thetalow + inttol & ...
                thetawant < thetahigh - inttol
         
         % lower midpoint
         ind = ind + 1;
         thetanew(ind,1) = (thetalow + thetawant)/2;
         
         % shift thetanew(ind) into the interval [-pi,pi] again
         if (thetanew(ind,1) >= 2*pi)             
             thetanew(ind,1) = thetanew(ind,1) - 2*pi;
         end         
         if (thetanew(ind,1) >= pi)
             thetanew(ind,1) = thetanew(ind,1) - 2*pi;
         end

         
         
         
         % upper midpoint
         ind = ind + 1;
         thetanew(ind,1) = (thetawant + thetahigh)/2;
         
         % shift thetanew(ind) into the interval [-pi,pi] again
         if (thetanew(ind,1) >= 2*pi)
             thetanew(ind,1) = thetanew(ind,1) - 2*pi;
         end
         if (thetanew(ind,1) >= pi)
             thetanew(ind,1) = thetanew(ind,1) - 2*pi;
         end

     else
         % otherwise, if thetawant is not in the interval
         % take the midpoint of thetalow and thetahigh
         ind = ind + 1;
         thetanew(ind,1) = (thetalow + thetahigh)/2;   
         
         % shift thetanew(ind) into the interval [-pi,pi] again
         if (thetanew(ind,1) >= 2*pi)
             thetanew(ind,1) = thetanew(ind,1) - 2*pi;
         end
         if (thetanew(ind,1) >= pi)
             thetanew(ind,1) = thetanew(ind,1) - 2*pi;
         end

                  
      end
   end
   
   
   
   
   if plotfig > 0
      % plot the circle with radius r
      plotcircle(r,plotfig);
      figure(plotfig);
      
      % plot the intersection points of the circle(with radius r) 
      % and the pseudo-spectrum
      pointsoncircle = r * (cos(theta) + i*sin(theta));
      plot(real(pointsoncircle), imag(pointsoncircle), 'g+')
      
  end
   
  % if A is real, discard the midpoints in the lower half plane
  if isreal(A)
     indx = find(thetanew >= 0);
     ynew = thetanew(indx);
  end

   
end
