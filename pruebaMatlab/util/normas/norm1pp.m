function gama=norm1pp(A,p,B)
%   gama=norm1pp(A,p,B)
%
%   Estimates the 1-norm of A^p*B, where A and B are square matrix of the 
%   same dimension.
%
%   Inputs:
%      A:  the input matrix A.
%      p:  power parameter.
%      B:  the input matrix B.
%
%   Outputs: 
%      gama: estimated 1-morm of A^p*B.
%
%   References: 
%   [1] FORTRAN Codes for Estimating the One-Norm of a Real or Complex
%       Matrix, with Applications to Condition Estimation.
%       ACM Transactions on Mathematical Software, Vol. 14, No. 4, 
%       pp. 381-396, 1988 (Algorithm 4.1).
%
%   Revised version 1.0, 16/05/2013

n=size(A,1);
v=B*(ones(n,1)/n);
for i=1:p
    v=A*v;
end
gama=norm(v,1);
psi=sign(v);
x=B'*psi;
for i=1:p
    x=A'*x;
end
k=2;fin=0;
while fin==0
	[~,j]=max(abs(x));
    ej=zeros(n,1);
    ej(j)=1;
    v=B*ej;
    for i=1:p
        v=A*v;
    end
    gama1=gama;
    gama=norm(v,1);
    aux=max(abs(sign(v)-psi));
    if aux==0||(gama<=gama1)
        break
    end
    psi=sign(v);
    x=B'*psi;
	for i=1:p
        x=A'*x;
    end
    k=k+1;
    fin=norm(x,Inf)==x(j)||k>5;
end
for i=1:n
    x(i)=(1+(i-1)/(n-1))*(-1)^(i+1);
end
x=B*x;
for i=1:p
	x=A*x;
end
aux=2*norm(x,1)/(3*n);
if aux>gama
    gama=aux;
end
end



