function [rbest, thetabest] = pspr_2way_rad(A, mE, theta, imagtol, plotfig, rold, iter,h)
% Michael Overton and Emre Mengi
% called by pspr_2way.m
% Given a set of angles in theta, in each direction finds the point 
% on the pseudospectrum boundary with the largest radius, i.e.
%           rnew(j) = max |z|
%                     s.t. sigma_min(A - z I) = epsln and angle(z) = theta(j)
% rbest is the maximum of the largest radius in any direction
%           rbest = max rnew(j)
%                 1<=j<=length(theta)
% thetabest is the angle of a point on the boudary with radius rbest in one of
% the directions in theta, i.e. let rbest = rnew(k), for some k, 1<=k<=length(theta),
% then thetabest = theta(k).


n = length(A);
for j=1:length(theta)
    
   % I*e^(i*theta(j))
   eith = (cos(theta(j)) + i*sin(theta(j))) * eye(n);
   % I*e^(-i*theta(j))
   emith = (cos(theta(j)) - i*sin(theta(j))) * eye(n);
   % null matrix of size n
   O = 0 * eye(n);
   
   % compute the generalized eigenvalues for the pencil H - lambda*K
   H = [mE A; A' mE];
   K = [O eith; emith O];
   eHK = eig(H,K);
    
     
   if min(abs(imag(eHK))) <= imagtol % check if pencil H-lambda*K has a real generalized eigenvalue
      indx = find(abs(imag(eHK)) <= imagtol);  % extract such eigenvalues
      rnew(j) = real(max(real(eHK(indx))));
      
      if plotfig > 0
        figure(plotfig)
        % plot a line between old radius and new radius        
        if (rnew(j) > rold)
            plot(rold*cos(theta(j)), rold*sin(theta(j)), 'bx')
            plot([rold*cos(theta(j)) rnew(j)*cos(theta(j))], [rold*sin(theta(j)) rnew(j)*sin(theta(j))], 'm-')
            % plot new radius
            plot(rnew(j)*cos(theta(j)), rnew(j)*sin(theta(j)), 'b+')
            
            
            if isreal(A)
                plot(rold*cos(theta(j)), -rold*sin(theta(j)), 'bx')
                plot([rold*cos(theta(j)) rnew(j)*cos(theta(j))], -[rold*sin(theta(j)) rnew(j)*sin(theta(j))], 'm-')
                % plot new radius
                plot(rnew(j)*cos(theta(j)), -rnew(j)*sin(theta(j)), 'b+')
            end
        end
                       
      end % end of if plotfig > 0
   else 
      % there may be no point on the boundary of the pseudospectrum
      % in the direction theta(j)
      rnew(j) = -inf;
   end % end of else
   
end % end of for

if isempty(rnew)
    if (n >= 30)
       close(h)
    end
    error('no intersection point is found by the radial search(please try smaller epsilon)')
end

% choose the maximum of the largest radius in directions included in theta
[rbest,ind] = max(rnew);
thetabest = theta(ind);

% mark the chosen point on the circle with radius rbest
if plotfig > 0
   plot(rbest*cos(thetabest), rbest*sin(thetabest), 'b*')
end
