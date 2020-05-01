function [fAv,m,s,np] = expmvhigham_inef(A,v)
[fA,m,s,np] = expm_newm(A,0);
fAv=fA*v;