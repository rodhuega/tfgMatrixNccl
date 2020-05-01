function ps_data = switch_dots(fig,cax,this_ver,ps_data)

% Funcion to plot eigenvalues of random perturbations to
% the input matrix onto the current plot.

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

n = length(ps_data.schur_matrix);

mtxs = input('Number of matrices? ');
ep = input('Size of perturbation? ');

ews = zeros(n*mtxs,1);

for i=1:mtxs,

  v1 = randn(n,1)+1i*randn(n,1); v1 = v1/norm(v1);
  v2 = randn(1,n)+1i*randn(1,n); v2 = v2/norm(v2);

  E = ep*v1*v2;

  ews((i-1)*n+1:i*n) = eig(ps_data.schur_matrix+E);
  
end;

plot(real(ews),imag(ews),'.','color',0.6*[1 0 1]);

ps_data.rand_pert_ews = ews;
