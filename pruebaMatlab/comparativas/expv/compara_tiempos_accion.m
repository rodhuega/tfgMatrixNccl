function compara_tiempos_accion
n=[64 128 256 512 1024 2048 4096];
for i=1:length(n)
    A=rand(n(i));
    v=rand(n(i),1);
    tic;
    [fT1,m,s,np] = fun_pol('exp','taylor','conEstNorma','sinGPUs',A);
    fT1=fT1*v;
    t1(i)=toc;
    tic;
    [fT2,m,s,np] = fun_pol('expv','taylor','conEstNorma','sinGPUs',A,v);
    t2(i)=toc;
    tic;
    error=norm(fT1-fT2)/norm(fT1);
    fprintf('Tamaño %dx%d. norm(fT1-fT2)/norm(fT1)= %e\n',n(i),n(i),error);
    [fH,s,m,mv,mvd,unA] = expmv(1,A,v,[],'double');
    t3(i)=toc;
end
plot(n,t1)
hold on
plot(n,t2)
plot(n,t3)
legend('Ineficiente','Eficiente','Higham');
xlabel('Tamaño');
ylabel('Tiempo (segs.)');
title('Tiempos acción exponencial');
[t1' t2' t3']
    
    