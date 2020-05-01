function [B,m,s,np] = expmspl(A, NormEst)
%EXPMSPL  Matrix exponential function.
%
%   [B,m,s,np] = expmspl(A, NormEst) computes the matrix exponential of A 
%   using a scaling and squaring algorithm with a matrix spline 
%   approximation [1]. 
%
%   Inputs:
%      A:    the input matrix
%       NormEst = 0 no estimation of norms of matrix powers is used.
%       NormEst = 1 estimation of norms of matrix powers is used.
%
%   Outputs: 
%      B:  the exponential of matrix A.
%      m:  the approximation order used.
%      s:  the scaling parameter.
%      np: Cost in terms of number of evaluations of matrix products.  
%
%   Reference:
%   [1] A new efficient and accurate spline algorithm for the matrix 
%       exponential computation, Journal of Computational and Applied 
%       Mathematics. In Press, https://doi.org/10.1016/j.cam.2017.11.029,
%       Elsevier, 2017.
%
%   Author: Javier Ibáñez
%   Revised version: 2018/02/6.
%
%   Group of High Performance Scientific Computing (HiPerSC)
%   Universitat Politecnica de Valencia (Spain)
%   http://hipersc.blogs.upv.es
%
if nargin<1 || nargin>2
    error('expmspl:NumParameters','Incorrect number of input arguments.');
elseif nargin==1
    NormEst=1;
end
if NormEst==1
    [m,s,pA,met,im]=select_m_s(A);
else
    [m,s,pA,met,im]=select_m_s_n(A);
end
if met==1
    [B,np]=taysplc_exp(pA,m,im);
    for k = 1:s
        B = B*B;  % Squaring
    end
    np=np+s;
else
    [B,np] = tayr_exp(pA,m);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [m,s,pA,met,im]=select_m_s(A)
% [m,s,pA,met,im]=select_m_s(A)
% Given a square matrix A, this function obtains the order m and scaling
% parameter s for Taylor/Spline approximation (maximum order of m is 30)
% Optimal orders m:            2 4 6 9 12 16 20 25 30
% Corresponding matrix powers: 2 2 3 3  4  4  5  5  6
% Estimation of norms of matrix powers is used
%
%   Input:
%      A: the input matrix.
%
%   Outputs:
%      m: order or approximation to be used.
%      s: scaling parameter
%      pA: cell array with the powers of A, A^2,A^3,...,A^maxpow.
%      met:
%         met=0, Taylor approximation for Nilpotent matrices is used
%         met=1, Taylor approximation for no Nilpotent matrices is used
%
% Theta from forward and backward error bounds from [1] Table 2 
Theta=[2.675298260329713e-8 %m=2  forward
       3.397168839977002e-4 %m=4  forward
       9.065656407595296e-3 %m=6  forward
       8.957760203223343e-2 %m=9  forward
       2.996158913811581e-1 %m=12 forward
       7.802874256626574e-1 %m=16 forward
       1.415070447561532    %m=20 backward
       2.353642766989427    %m=25 backward
       3.411877172556770];  %m=30 backward
met=1;
im=0;
s = 0;
pA{1}=A;a(1)=norm(A,1); 
if a(1)==0,m=0;met=0;return;end %Null matrix
pA{2}=A*A;a(2)=norm(pA{2},1);
if a(2)==0,m=1;met=0;return;end %Nilpotent matrix

alfa=a(2)^(1/2);
if alfa<=Theta(1)%m=2  forward
    m=2;im=1;
    return;
end

alfa=norm1p(pA{2},2)^(1/4);%m=4  forward
if alfa<=Theta(2)
    m=4;im=2;
    return;
end

% m = 6
pA{3}=pA{2}*pA{1};a(3)=norm(pA{3},1);
if a(3)==0,m=2;met=0;return;end %Nilpotent matrix

alfa=norm1p(pA{3},2)^(1/6);%m=6  forward
if alfa<=Theta(3)
    m=6;im=3;
    return;
end
% m = 9
alfa=norm1p(pA{3},3)^(1/9);%m=9  forward
if alfa<=Theta(4)
    m=9;im=4;
    return;
end

% m = 12
pA{4}=pA{2}*pA{2};a(4)=norm(pA{4},1);
if a(4)==0,m=3;met=0;return;end %Nilpotent matrix
alfa=norm1p(pA{4},3)^(1/12);%m=12  forward
if alfa<=Theta(5)
    m=12;im=5;
    return;
end
% m = 16
alfa=norm1p(pA{4},4)^(1/16);%m=16  forward
if alfa<=Theta(6)
    m=16;im=6;
    return;
end
% m = 20
pA{5}=pA{4}*pA{1};a(5)=norm(pA{5},1);
if a(5)==0,m=4;met=0;return;end %Nilpotent matrix
alfa=norm1p(pA{5},4)^(1/20);%m=20  backward
if alfa<=Theta(7)
    m=20;im=7;
    return;
end

alfa25=norm1p(pA{5},5)^(1/25);%m=25  backward
if alfa25<=Theta(8)
    m=25;im=8;
    return;
end

%pA{6}=pA{1}*pA{5};
pA{6}=pA{5}*pA{1};
a(6)=norm(pA{6},1);
if a(6)==0,m=5;met=0;return;end %Nilpotent matrix
alfa=norm1p(pA{6},5)^(1/30);%m=30  backward
if alfa<=Theta(9)
    m=30;im=9;
    return;
end
s= ceil(log2(alfa/Theta(9)));
s25=ceil(log2(alfa25/Theta(8)));
if s25+1>=s
    m=30;im=9;
else
    m=25;im=8;
	s=s25;    
end
for i=1:length(pA)
    pA{i}=pA{i}/2^(s*i);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [m,s,pA,met,im]=select_m_s_n(A)
% [m,s,pA,met,im]=select_m_s_n(A)
% Given a square matrix A, this function obtains the order m and scaling
% parameter s for Taylor/Spline approximation (maximum order of m is 30)
% Optimal orders m:            2 4 6 9 12 16 20 25 30
% Corresponding matrix powers: 2 2 3 3  4  4  5  5  6
% No estimation of norms of matrix powers is used
%
%   Input:
%      A: the input matrix.
%
%   Outputs:
%      m: order or approximation to be used.
%      s: scaling parameter
%      pA: cell array with the powers of A, A^2,A^3,...,A^maxpow.
%      met:
%         met=0, Taylor approximation for Nilpotent matrices is used
%         met=0, Taylor approximation for no Nilpotent matrices is used
%
% Theta from forward and backward error bounds from [1] Table 2 
Theta=[2.675298260329713e-8 %m=2  forward
       3.397168839977002e-4 %m=4  forward
       9.065656407595296e-3 %m=6  forward
       8.957760203223343e-2 %m=9  forward
       2.996158913811581e-1 %m=12 forward
       7.802874256626574e-1 %m=16 forward
       1.415070447561532    %m=20 backward
       2.353642766989427    %m=25 backward
       3.411877172556770];  %m=30 backward
met=1;
%Initial scaling parameter s = 0 
s = 0;
% m = 1
pA{1}=A; a1=norm(pA{1},1);
if a1==0,m=0;met=0;return;end %Null matrix
pA{2}=A*A;a2=norm(pA{2},1);
if a2==0,m=1;met=0;return;end %Nilpotent matrix
% m = 2
alfa=a2^(1/2);
if alfa<=Theta(1)%m=2  forward
    m=2;im=1;
    return;
end

% m = 4
a4=a2^2;alfa=a4^(1/4);
if alfa<=Theta(2)%m=4  forward
    m=4;im=2;
    return;
end

% m = 6
pA{3}=pA{2}*pA{1};a3=norm(pA{3},1);
if a3==0,m=2;met=0;return;end %Nilpotent matrix
a6=min([a3^2,a2^3]);
alfa=a6^(1/6);
if alfa<=Theta(3)%m=6  forward
    m=6;im=3;
    return;
end
% m = 9
a9=a3*a6;
alfa=a9^(1/9);
if alfa<=Theta(4)%m=9  forward
    m=9;im=4;
    return;
end

% m = 12
pA{4}=pA{2}*pA{2};a4=norm(pA{4},1);
if a4==0,m=3;met=0;return;end %Nilpotent matrix
a12=min([a3*a9,a4*a6*a2,a4^3]);
alfa=a12^(1/12);
if alfa<=Theta(5)%m=12  forward
    m=12;im=5;
    return;
end
% m = 16
a16=a4*a12;
alfa=a16^(1/16);
if alfa<=Theta(6)%m=16  forward
    m=16;im=6;
    return;
end
% m = 20
pA{5}=pA{4}*pA{1};a5=norm(pA{5},1);
if a5==0,m=4;met=0;return;end %Nilpotent matrix
a20=min([a4*a16,a5*a12*a3,a5^2*a9*a1,a5^2*a4^2*a2,a5^4]);
alfa=a20^(1/20);
if alfa<=Theta(7)%m=20 backward
    m=20;im=7;
    return;
end
% m = 25
a25=a5*a20;
alfa25=a25^(1/25);
if alfa25<=Theta(8)%m=25 backward
    m=25;im=8;
    return;
end
% m = 30
pA{6}=pA{5}*pA{1};a6=norm(pA{6},1);
if a6==0,m=5;met=0;return;end %Nilpotent matrix

a30=min([a5*a25,a6^5,a6^3*a12,a6^3*a5^2*a2,a6^2*a16*a2,a6^2*a5^3*a3,a6^2*a5^2*a4^2,a6*a20*a4]);
alfa=a30^(1/30);
if alfa<=Theta(9)%m=30 backward
    m=30;im=9;
    return;
end
s=ceil(log2(alfa/Theta(9)));
s25=ceil(log2(alfa25/Theta(8)));
if s25+1>=s
    m=30;im=9;
else
    m=25;im=8;
	s=s25;    
end
for i=1:length(pA)
    pA{i}=pA{i}/2^(s*i);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [B,np]=taysplc_exp(pA,m,im)
%   [B,np]=taysplc_exp(pA,m,im)
%   Computes the exponential of matrix A by a matrix spline method from [1].
%   Inputs:
%     pA: cell array with the powers of A nedeed to compute Taylor
%          series, such that Ap{i} contains A^i, for i=2,3,...,q.
%      m:  order or approximation to be used.
%     im:  index of Q: Q(im)=maximum matrix power used (A,A^2,...,A^q)
%          m: 2 4 6 9 12 16 20 25 30
%          q: 2 2 3 3  4  4  5  5  6
%
%   Outputs:
%      B:  the exponential of matrix A.
%      np: matrix products carried out by the function.
%
Q=[2, 2, 3, 3, 4, 4, 5, 5, 6];
q=Q(im);
n = size(pA{1},1);
max_cond=1e1;
p =[ 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000, 2432902008176640000, 51090942171709440000, 1124000727777607680000, 25852016738884978212864, 620448401733239409999872, 15511210043330986055303168, 403291461126605650322784256, 10888869450418351940239884288, 304888344611713836734530715648, 8841761993739700772720181510144, 265252859812191032188804700045312, 8222838654177922430198509928972288, 263130836933693517766352317727113216, 8683317618811885938715673895318323200, 295232799039604119555149671006000381952, 10333147966386144222209170348167175077888, 371993326789901177492420297158468206329856, 13763753091226343102992036262845720547033088, 523022617466601037913697377988137380787257344, 20397882081197441587828472941238084160318341120, 815915283247897683795548521301193790359984930816, 33452526613163802763987613764361857922667238129664, 1405006117752879788779635797590784832178972610527232, 60415263063373834074440829285578945930237590418489344, 2658271574788448529134213028096241889243150262529425408, 119622220865480188574992723157469373503186265858579103744, 5502622159812088456668950435842974564586819473162983440384, 258623241511168177673491006652997026552325199826237836492800, 12413915592536072528327568319343857274511609591659416151654400];
I = eye(n);
B=m*pA{1}-I;
nc=condest(B);
if nc<max_cond
    B=B\pA{q}/p(m-1);
    np=im+4/3;
else
    B=pA{q}/p(m);
    np=im;
end
switch m
    case 2 %q=2
        B= B +pA{1} + I;
	case 4 %q=2
        B=B+ pA{1}/p(3) + I/p(2);
        B=B*pA{2}               + pA{1}      + I;
	case 6 %q=3
        B=B       + pA{2}/p(5) + pA{1}/p(4) + I/p(3);
        B=B*pA{3} + pA{2}/p(2)  + pA{1}      + I;
	case 9 %q=3
        B=B       + pA{2}/p(8) + pA{1}/p(7) + I/p(6);
        B=B*pA{3} + pA{2}/p(5)  + pA{1}/p(4) + I/p(3);
        B=B*pA{3} + pA{2}/p(2)  + pA{1}      + I;
	case 12 %q=4
        B=B       + pA{3}/p(11) + pA{2}/p(10)+ pA{1}/p(9) + I/p(8);
        B=B*pA{4} + pA{3}/p(7)  + pA{2}/p(6)  + pA{1}/p(5) + I/p(4);
        B=B*pA{4} + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}      + I;
	case 16 %q=4
        B=B       + pA{3}/p(15) + pA{2}/p(14) + pA{1}/p(13) + I/p(12);
        B=pA{4}*B + pA{3}/p(11) + pA{2}/p(10) + pA{1}/p(9)  + I/p(8);
        B=pA{4}*B + pA{3}/p(7)  + pA{2}/p(6)  + pA{1}/p(5)  + I/p(4);
        B=pA{4}*B + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}       + I;
	case 20 %q=5
        B=B       + pA{4}/p(19) + pA{3}/p(18) + pA{2}/p(17) + pA{1}/p(16) + I/p(15);
        B=pA{5}*B + pA{4}/p(14) + pA{3}/p(13) + pA{2}/p(12) + pA{1}/p(11) + I/p(10);
        B=pA{5}*B + pA{4}/p(9)  + pA{3}/p(8)  + pA{2}/p(7)  + pA{1}/p(6)  + I/p(5);
        B=pA{5}*B + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I;
	case 25 %q=5
        B=B       + pA{4}/p(24) + pA{3}/p(23) + pA{2}/p(22) + pA{1}/p(21) + I/p(20);
        B=pA{5}*B + pA{4}/p(19) + pA{3}/p(18) + pA{2}/p(17) + pA{1}/p(16) + I/p(15);
        B=pA{5}*B + pA{4}/p(14) + pA{3}/p(13) + pA{2}/p(12) + pA{1}/p(11) + I/p(10);
        B=pA{5}*B + pA{4}/p(9)  + pA{3}/p(8)  + pA{2}/p(7)  + pA{1}/p(6)  + I/p(5);
        B=pA{5}*B + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I;
	case 30 %q=6
        B=B        + pA{5}/p(29) + pA{4}/p(28) + pA{3}/p(27) + pA{2}/p(26) + pA{1}/p(25) + I/p(24);
        B=pA{6}*B  + pA{5}/p(23) + pA{4}/p(22) + pA{3}/p(21) + pA{2}/p(20) + pA{1}/p(19) + I/p(18);
        B=pA{6}*B  + pA{5}/p(17) + pA{4}/p(16) + pA{3}/p(15) + pA{2}/p(14) + pA{1}/p(13) + I/p(12);
        B=pA{6}*B  + pA{5}/p(11) + pA{4}/p(10) + pA{3}/p(9)  + pA{2}/p(8)  + pA{1}/p(7)  + I/p(6);
        B=pA{6}*B  + pA{5}/p(5)  + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I;
end
end

function [B,np]= tayr_exp(pA,m)
%[B,np]= tayr_exp(Ap,m)
%   Computes the exponential for nilpotent matrices  by Taylor
%   approximations.
%   Inputs:
%     pA: cell array with the powers of A nedeed to compute Taylor
%          series, such that pA{i} contains A^i, for i=2,3,...,q.
%      m:  order or approximation to be used.
%
%   Outputs:
%      B:  the exponential of matrix A.
%      np: matrix products carried out by the function.
%
%
n = size(pA{1},1); 
I = eye(n);
p =[ 1, 2, 6, 24];
switch m 
    case 0
        B=I;
        np=0;
	case 1
        B= pA{1}   + I;
        np=0;
	case 2 
        B=pA{2}/p(2) + pA{1}  + I;
        np=1;
    case 3  
        np=2;
        B=(pA{1}/p(3)+I/p(2))*pA{2} + pA{1} + I;
	case 4
        np=2;
        B=pA{2}/p(4) + pA{1}/p(3)   + I/p(2);
        B=B*pA{2}    + pA{1}   + I;
        
           B=B+ pA{1}/p(3) + I/p(2);
        B=B*pA{2}               + pA{1}      + I;     
        
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

end
