function serie=series_back_taylor(f,m,n)
%e_backward=coef_back(f,m,itermax)
%This function computes  the first  aproximately n terms of the
%corresponding series of the backward error of function f(x).
%Input Data
%  f: Analytic function.
%  m: Positive integer equal to the order of the Taylor approximation.
%  n: Positive integer. The number of the computed terms of backward error 
%     is equal to the first multiple of m+1 greater or equal to n.
%Output Data
%  e_backward: vector with aproximately the first n coefficients of 
%     backward error.
%Example 1: MATLAB Built-in functions
%e_backward=coef_back(@exp,4,100)
%
%Example 2: General MATLAB  functions
%If we define the following functions as a MATLAB function
% function f= ej(x)
% f=sin(x)*exp(x);
% end
%Then
%e_backward=coef_back(@ej,4,100)

syms x
Tmc=taylor(f(x),'order',m+1);
c = sym('c',[1,m+1]);
mlow=m+1;
mhigh=mlow+m;
DeltaX=c*(x.^(m+1:mhigh)).';
Tmmc=taylor(f(x+DeltaX)-Tmc,'order',mhigh+1);
[csolve,~]=coeffs(Tmmc,x);
lcsolve=length(csolve);
e_backward=subs(c(1:lcsolve),solve(csolve,c));
itermax=ceil(n/lcsolve);
for k=2:itermax
    mlow=mlow+lcsolve;
    mhigh=mhigh+lcsolve;
    DeltaX=[e_backward(1:mlow-1-m),c]*(x.^(m+1:mhigh)).';
    Tmmc=taylor(f(x+DeltaX)-Tmc,'order',mhigh+1);
    [csolve,~]=coeffs(Tmmc,x);
    e_backward((mlow:mhigh)-m)=subs(c,solve(csolve,c));
end
if lcsolve<m+1
    e_backward=e_backward(1:end-1); 
end
lb=length(e_backward);
serie=e_backward*(x.^(m+1:m+lb)).';
end
