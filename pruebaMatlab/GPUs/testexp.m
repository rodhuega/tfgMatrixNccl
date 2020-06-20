function [t_cpu_tay,t_gpu_tay,t_cpu_ber,t_gpu_ber] = testexp(n_ini,inc,n_end);
% [t_cpu_tay,t_gpu_tay,t_cpu_ber,t_gpu_ber] = testexp(n_ini,inc,n_end);
%
% Funcion utilizada para la comparativa del artículo JCAM del CMMSE2019

% Dummy entries to load GPU driver
A=rand(100);
[fA,m,s,nProd] = fun_pol('exp','bernoulli','conEstNorma','sinGPUs',A);
[fB,m,s,nProd] = fun_pol('exp','bernoulli','conEstNorma','conGPUs',A);
[fB,m,s,nProd] = fun_pol('exp','bernoulli','conEstNorma','conGPUs',A);

ind_t = 1;
for i=n_ini:inc:n_end
  A=rand(i);
  A=A/norm(A);%*500;

  % tic
  % [fA,m,s,np] = fun_pol('exp','taylor','conEstNorma','sinGPUs',A);
  % t = toc;
  % t_cpu_tay(ind_t) = t;

  % tic
  % [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNorma','sinGPUs',A);
  % t = toc;
  % t_cpu_ber(ind_t) = t;

  % tic
  % [fB,m,s,np] = fun_pol('exp','taylor','conEstNorma','conGPUs',A);
  % t = toc;
  % t_gpu_tay(ind_t) = t;
  %[fB,m,s,np] = fun_pol('exp','bernoulli','sinEstNormaSplines','conGPUs',A);


  tic
  [fB,m,s,np] = fun_pol('exp','bernoulli','conEstNorma','sinGPUs',A);
  t = toc;
  t_gpu_ber(ind_t) = t;
  m_ber(ind_t) = m;
  ind_t=ind_t+1;
  %disp(norm(fA-fB)/norm(fA));
end
disp('Tiempo');
disp(t_gpu_ber);

disp('M');
disp(m_ber);