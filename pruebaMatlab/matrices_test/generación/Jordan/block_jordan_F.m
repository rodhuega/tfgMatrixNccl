function [J,FJ]=block_jordan_F(F,a,n,nd)
syms x;
digits(nd);
a=vpa(a);
Fa=vpa(F(vpa(a)));
J=diag(a*ones(n,1))+diag(ones(n-1,1),1);
FJ=diag(Fa*ones(n,1));
Fs=F(x);
for k=2:n
    f=vpa(subs(diff(Fs,k-1),a))/vpa(factorial(k-1));
    FJ=vpa(FJ+diag(f*ones(n-k+1,1),k-1));
end

