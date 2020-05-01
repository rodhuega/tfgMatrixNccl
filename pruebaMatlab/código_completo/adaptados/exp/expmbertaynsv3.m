function [F,m,s,Cp] = expmbertaynsv3(A,kmax)
%expmbertaynsv3  Matrix exponential function.
%
%   expmbertaynsv3(A, kmax) computes the matrix exponential of A using a scaling 
%   and squaring algorithm and Taylor or Bernoulli approximation.
%
%   Inputs:
%      A:    the input matrix
%      kmax: a vector index to determine the maximum approximation order 
%            'm' to be used: 
%
%                                    kmax:  6  7  8  9
%            Max. approximation order (m): 16 20 25 30
%
%            If kmax<6, kmax is set to 6. If kmax>9 or kmax is not
%            provided, it is set to 9 (recommended value).
%
%   Outputs: 
%      F:  the exponential of matrix A.
%      m:  the approximation order used.
%      s:  the scaling parameter.
%      Cp: Cost in terms of number of evaluations of matrix products.  
%
%   References:
%
%   [1] P. Ruiz, J. Sastre, J. Ibáñez, E. Defez, High perfomance computing 
%       of the matrix exponential, Journal of Computational and Applied 
%       Mathematics, 291, 2016, 370–379.
%
%   [2] P. Ruiz, J. Sastre, J. Ibáñez, E. Defez, A new efficient algorithm for 
%       matrix exponential computation, in Proc. of the International Conference 
%       on Mathematical Modelling for Engineering and Human Behaviour 2014 
%       (I.S.B.N.: 978-84-606-5746-0), Valencia, September 3th-5th, pp. 128-133
%
%   Authors: Pedro A. Ruiz, Jorge Sastre and Javier Ibáñez.
%   Revised version: 2017/10/23.
%
%   Group of High Performance Scientific Computation (HiPerSC)
%   Universitat Politecnica de Valencia (Spain)
%   http://hipersc.blogs.upv.es

% Check arguments
minkmax = 6;  %Minimum considered order m_M = 16
maxkmax = 9;  %Maximum considered order m_M = 30
if nargin<1 || nargin>2
    error('Incorrect number of input arguments.');
elseif nargin==1
    kmax = maxkmax;
elseif nargin==2
    if kmax<minkmax
        kmax = minkmax;
    elseif kmax>maxkmax
        kmax = maxkmax;
    end
end

Thetam1=1.490116111983279e-8; %m=1
theta=[8.733457513635361e-006 %m=2
       1.678018844321752e-003 %m=4
       1.773082199654024e-002 %m=6
       1.137689245787824e-001 %m=9
       3.280542018037257e-001 %m=12
       7.912740176600240e-001 %m=16
       1.438252596804337      %m=20
       2.428582524442827      %m=25
       3.539666348743690];    %m=30

c=[   4/3   %m=2   %c=abs(c_m(m+1)/c_m(m+2))
      6/5   %m=4   
      8/7   %m=6
    11/10   %m=9
    14/13   %m=12
    18/17   %m=16
    22/21   %m=20
    27/26   %m=25
    32/31]; %m=30

ucm2= [8.9e-016                %m=2 
       1.598721155460225e-014  %m=4 
       6.394884621840902e-013  %m=6
       4.431655042935745e-010  %m=9
       7.445180472132051e-007  %m=12
       4.181213353149360e-002  %m=16
       5.942340417495871e+003  %m=20
       4.649643683073819e+010  %m=25
       9.423674633957229e+017];%m=30

mv = [2 4 6 9 12 16 20 25 30];  %Vector of optimal orders m.

if mv(kmax)>20 %m_M = 25 or 30
    qv = [2  2  3  3  4  4  5  5  5];
else           %m_M = 16 or 20
    qv = [2  2  3  3  4  4  4];
end

n = size(A,1);
s = 0;
Cp = 0;
mmax = mv(kmax);
qmax = qv(kmax);
a=-ones(1,mmax+2);
a(1) = norm(A,1);
if a(1)<=Thetam1            % m=1 not very likely to be used
    F = A + eye(n);                   
    m = 1;                            
    return
end

Ap = cell(qv(kmax),1);
Ap{1}=A;

for inm=1:kmax
    m = mv(inm);
    q = qv(inm);
    switch m
        case 2
            Ap{2} = A*A;
            Cp = Cp+1;
            a(3) = norm1pp(Ap{2},1,A);
            b = max(1,a(1))*ucm2(1);
            if c(1)*a(3)<=b
                a(4) = norm1p(Ap{2},2);
                if c(1)*a(3)+a(4)<=b 
                    [F, p] = MTPS(A,Ap,m,q,s);
                    Cp = Cp+p;
                    return
                end
            end

        case 4
            a(5) = norm1pp(Ap{2},2,A);
            b = max(1,a(1))*ucm2(2);
            if c(2)*a(5)<=b
                a(6) = norm1p(Ap{2},3);
                if c(2)*a(5)+a(6)<=b 
                    [F, p] = MTPS(A,Ap,m,q,s);
                    Cp = Cp+p;
                    return
                end
            end
            
        case 6
            Ap{3} = Ap{2}*A;
            Cp = Cp+1;
            a(7) = norm1pp(Ap{3},2,A);  
            b = max(1,a(1))*ucm2(3);
            if c(3)*a(7)<=b
                a(8) = norm1pp(Ap{3},2,Ap{2});
                if c(3)*a(7)+a(8)<=b 
                    [F, p] = MTPS(A,Ap,m,q,s);
                    Cp = Cp+p;
                    return
                end
            end
            
        case 9
            a(10) = norm1pp(Ap{3},3,A);
            b = max(1,a(1))*ucm2(4);
            if c(4)*a(10)<=b
                a(11) = norm1pp(Ap{3},3,Ap{2});
                if c(4)*a(10)+a(11)<=b 
                    [F, p] = MTPS(A,Ap,m,q,s);
                    Cp = Cp+p;
                    return
                end
            end
            
        case 12
            %Ap{4} = Ap{2}*Ap{2};
            Ap{4} = Ap{3}*A;
            Cp = Cp+1;
            a(13) = norm1pp(Ap{4},3,A);  
            b = max(1,a(1))*ucm2(5);
            if c(5)*a(13)<=b
                a(14) = norm1pp(Ap{4},3,Ap{2});
                if c(5)*a(13)+a(14)<=b 
                    [F, p] = MTPS(A,Ap,m,q,s);
                    Cp = Cp+p;
                    return
                end
            end
            
        case 16
            a(17) = norm1pp(Ap{4},4,A);
            b = max(1,a(1))*ucm2(6);
            if c(6)*a(17)<=b
                a(18) = norm1pp(Ap{4},4,Ap{2});
                if c(6)*a(17)+a(18)<=b 
                    [F, p] = MTPS(A,Ap,m,q,s);
                    Cp = Cp+p;
                    return
                end
            end
            
        case 20
            Ap{5} = Ap{4}*A;
            Cp = Cp+1;
            a(21) = norm1pp(Ap{5},4,A);  
            b = max(1,a(1))*ucm2(7);
            if c(7)*a(21)<=b
                a(22) = norm1pp(Ap{5},4,Ap{2});
                if c(7)*a(21)+a(22)<=b 
                    [F, p] = MTPS(A,Ap,m,q,s);
                    Cp = Cp+p;
                    return
                end
            end

        case 25
            a(26) = norm1pp(Ap{5},5,A);  
            b = max(1,a(1))*ucm2(8);
            if c(8)*a(26)<=b
                a(27) = norm1pp(Ap{5},5,Ap{2});
                if c(8)*a(26)+a(27)<=b 
                    %[F, p] = MTPS(A,Ap,m,q,s);
                    [F,p]=calcula_fun_pol('exp','bernoulli','sinGPUs',Ap,[],m,s);
                    Cp = Cp+p;
                    return
                end
            end
            
        case 30
            a(31) = norm1pp(Ap{5},6,A);  
            b = max(1,a(1))*ucm2(9);
            if c(9)*a(31)<=b
                a(32) = norm1pp(Ap{5},6,Ap{2});
                if c(9)*a(31)+a(32)<=b 
                    %[F, p] = MTPS(A,Ap,m,q,s);
                    [F,p]=calcula_fun_pol('exp','bernoulli','sinGPUs',Ap,[],m,s);
                    Cp = Cp+p;
                    return
                end
            end         
    end
end

% Estimate 1-norm of A^(mmax+2) if it is not already estimated
if a(mmax+2)<0
    a(mmax+2) = norm1pp(Ap{qmax},floor((mmax+2)/qmax),Ap{mod((mmax+2),qmax)});
end

% Compute alpha_min
normA = max(a(mmax+1)^(1/(mmax+1)), a(mmax+2)^(1/(mmax+2)));
[t0 s] = log2(normA/theta(kmax));
s = s - (t0 == 0.5);

% Test if s can be reduced, checking if (5) of [2] holds with s=s-1
if s>0 
    s = s-1;
    b = max(1,a(1)/2^s)*ucm2(kmax);
    if c(kmax)*a(mmax+1)/2^((mmax+1)*s)+a(mmax+2)/2^((mmax+2)*s)>b
        % Cannot reduce s
        s = s+1;
    end
end

% Test if scaled matrix allows using mv(kmax-1)
if mmax>=20
    mmax = mv(kmax-1);
    qmax = qv(kmax-1);
    
    % Estimate 1-norm of A^(mmax+2) if it is not already estimated
    if a(mmax+2)<0
        a(mmax+2) = norm1pp(Ap{qmax},floor((mmax+2)/qmax),Ap{mod((mmax+2),qmax)});
    end
    
    b = max(1,a(1)/2^s)*ucm2(kmax-1);
    if c(kmax-1)*a(mmax+1)/2^((mmax+1)*s)+a(mmax+2)/2^((mmax+2)*s)<=b
        m = mmax;  % Scaled matrix allows using mv(kmax-1)
    end
end

if m<25
    [F, p] = MTPS(A,Ap,m,q,s);
else
    [F,p]=calcula_fun_pol('exp','bernoulli','sinGPUs',Ap,[],m,s);
end
Cp = Cp+p;


function [F, np] = MTPS(A,Ap,m,q,s)
%   [F, np] = MTPS(A,Ap,m,q,s)
%   Modified Taylor Paterson-Stockmeyer algorithm evaluating (7) of [1].
%   Computes the exponential of matrix A by means Taylor series.
%
%   Inputs:
%      A:  the input matrix.
%      Ap: cell array with the powers of A nedeed to compute Taylor 
%          series, such that Ap(i) contains A^i, for i=2,3,...,q.
%      m:  the order or approximation to be used.
%      q:  highest power if A computed and saved in Ap.
%      s:  scaling parameter.
%
%   Outputs: 
%      F:  the exponential of matrix A.
%      np: matrix products carried out by the function.  
%
%   Revised version 1.0, 16/05/2013

np = 0;
n = size(A);
if s
    s2=2^s;
    F=Ap{q}/(s2*m);
    for i=1:(q-2)
        F=(F+Ap{q-i})/(s2*(m-i));
    end
    F=F+A+(s2*(m-q+1))*eye(n);
    for j=1:(m/q-2)
        F=F*Ap{q}/(s2^2*(m-j*q+1)*(m-j*q));
        np=np+1;
        for i=1:(q-2)
            F=(F+Ap{q-i})/(s2*(m-j*q-i));
        end
        F=F+A+(s2*(m-(j+1)*q+1))*eye(n);
    end
    F=F*Ap{q}/(s2^2*(q+1)*q);
    np=np+1;
    for i=(q-1):-1:2
        F=(F+Ap{i})/(s2*i);
    end
    F=(F+A)/s2+eye(n);
    for i = 1:s % Squaring
        F = F*F;
    end
    np=np+s;
else
    F=Ap{q}/m;
    for i=1:(q-2)
        F=(F+Ap{q-i})/(m-i);
    end
    F=F+A+(m-q+1)*eye(n);
    for j=1:(m/q-2)
        F=F*Ap{q}/((m-j*q+1)*(m-j*q));
        np=np+1;
        for i=1:(q-2)
            F=(F+Ap{q-i})/(m-j*q-i);
        end
        F=F+A+(m-(j+1)*q+1)*eye(n);
    end
    F=F*Ap{q}/((q+1)*q);
    np=np+1;
    for i=(q-1):-1:2
        F=(F+Ap{i})/i;
    end
    F=F+A+eye(n);
end

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

function gama=norm1p(A,p)
%   gama=norm1p(A,p)
%
%   Estimates the 1-norm of A^p, where A is a square matrix.
%
%   Inputs:
%      A:  the input matrix
%      p:  power parameter
%
%   Outputs: 
%      gama: estimated 1-morm of A^p
%
%   References: 
%   [1] FORTRAN Codes for Estimating the One-Norm of a Real or Complex
%       Matrix, with Applications to Condition Estimation.
%       ACM Transactions on Mathematical Software, Vol. 14, No. 4, 
%       pp. 381-396, 1988 (Algorithm 4.1).
%
%   [2] norm1pp function.
%
%   Revised version 1.0: 16/05/2013

n=size(A,1);
v=A*(ones(n,1)/n);
for i=1:p-1
    v=A*v;
end
gama=norm(v,1);
psi=sign(v);
x=A'*psi;
for i=1:p-1
    x=A'*x;
end
k=2;fin=0;
while fin==0
	[~,j]=max(abs(x));
    ej=zeros(n,1);
    ej(j)=1;
    v=A*ej;
    for i=1:p-1
        v=A*v;
    end
    gama1=gama;
    gama=norm(v,1);
    aux=max(abs(sign(v)-psi));
    if aux==0||(gama<=gama1)
        break
    end
    psi=sign(v);
    x=A'*psi;
	for i=1:p-1
        x=A'*x;
    end
    k=k+1;
    fin=norm(x,Inf)==x(j)||k>5;
end
for i=1:n
    x(i)=(1+(i-1)/(n-1))*(-1)^(i+1);
end
x=A*x;
for i=1:p-1
	x=A*x;
end
aux=2*norm(x,1)/(3*n);
if aux>gama
    gama=aux;
end


