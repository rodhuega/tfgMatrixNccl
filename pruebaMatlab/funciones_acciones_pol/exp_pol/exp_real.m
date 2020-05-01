function [A,v,fAvreal]=exp_real(n)
rng('default');
d=rand(n,1);
A=diag(d);
v=rand(n,1)-0.5;
expd=vpa(exp(vpa(d)));
fA=diag(expd);
fAvreal=vpa(fA*v);


