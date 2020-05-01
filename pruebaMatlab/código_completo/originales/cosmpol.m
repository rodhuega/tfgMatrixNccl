function [A,m,s,nprods] = cosmpol(A,NormEst)
%   COSMPOL  Matrix cosine function based on Taylor series with new matrix polynomial evaluation algorithms.
%
%   cosmpol(A,NormEst) computes the matrix cosine cos(A) using a scaling 
%   and squaring algorithm and Taylor approximation with maximum order equal to 30.
%   Normest is an input that indicates if estimation norms of matrix powers are used.
%   If NormEst=0, then no estimation of norms of matrix powers are used,
%   else estimation of norms of matrix powers are used.
%   Function returns the matrix cosine, A; the appoximation
%   order used m; the scaling parameter s; and nprods in terms of 
%   number of matrix product evaluations.
%
%   References:
%   
%   [1] J. Sastre, J. Ibáñez, P. Alonso, J. Peinado, E. Defez, Fast Taylor polynomial 
%       evaluation for the computation of the matrix cosine, J. Comput. Appl. Math, 
%       in Print.
%
%   Author: Jorge Sastre.
%   Revised version: 2016/04/22.
%
%   Group of High Performance Scientific Computation (HiPerSC)
%   Universitat Politecnica de Valencia (Spain)
%   http://hipersc.blogs.upv.es
%  

%% Check arguments
if nargin<1 || nargin>2
    error('cosmpol:NumParameters','Incorrect number of input arguments.');
end
if nargin==1
    NormEst=1;
end


% Selection of optimal order m and scaling s
if NormEst==1
    [m,s,q,Bp,nprods] = ms_selectNormEst(A);
elseif NormEst==0
    [m,s,q,Bp,nprods] = ms_selectNoNormEst(A);
end

% Scaling
if s
    for i=1:q
        Bp{i}=Bp{i}/4^(s*i);
    end
end
% Evaluation of Taylor matrix polynomial by Paterson-Stockmeyer method
[A,npPOL]=cos_POL(Bp,m,q);
% Recovering cos(A) from the scaled aproximation cos(2^(-s)*A)
I=eye(size(A));
for i=1:s
    A=2*A*A-I;
end
% Total number of matrix products
nprods=nprods+npPOL+s;
end

function [m,s,q,pB,nprods]=ms_selectNormEst(A)
% m: 1 2 4 8 12 15
% q: 1 2 2 2  3  3
% Theta from forward and backward error bounds
Theta=[5.161913593731081e-08      %m=1  forward bound
       4.307691256676447e-05      %m=2  forward bound
       1.319680929892753e-02      %m=4  forward bound
       0.93933580333673539   %m=8  relative backward bound
       6.7523490074059511    %m=12 relative backward bound
       16.451238315562541];  %m=15 absolute forward bound Backward:9.9323454917022698 (1500 term, 150 vpa)


d=realmax*ones(1,17); %||B^k||, B=A^2
b=realmax*ones(1,17); %||B^k||^(1/k), B=A^2
%Initial scaling parameter s = 0 
s = 0;
% m = 1
pB{1}=A*A; nprods=1; q=1; d(1)=norm(pB{1},1);b(1)=d(1);
if d(1)<=Theta(1), m=1; return; end
% m = 2
pB{2}=pB{1}^2; nprods=2; q=2; d(2)=norm(pB{2},1);b(2)=d(2)^(1/2);
if d(2)==0, m=1; q=1; return; end %Nilpotent matrix
beta_min = beta_NoNormEst(b,d,2,q);
if beta_min<=Theta(2), m=2; return; end
% m = 4
beta_min = min(beta_min,beta_NoNormEst(b,d,4,q));
if beta_min<=Theta(3),
    d(3)=norm1pp(pB{2},1,pB{1});b(3)=d(3)^(1/3); %Test if previous m = 2 is valid
    if b(3)<=Theta(2)
        d(4)=min(d(2)^2,d(3)*d(1));b(4)=d(4)^(1/4);
        if b(4)<=Theta(2), m=2; return; end
        d(4)=norm1p(pB{2},2);b(4)=d(4)^(1/4);
        if b(4)<=Theta(2), m=2; return; end
    end
    m=4; return;
end
% m = 8
beta_min = min(beta_min,beta_NoNormEst(b,d,8,q));
if beta_min<=Theta(4)
    d(5)=norm1pp(pB{2},2,pB{1});b(5)=d(5)^(1/5); %Test if previous m = 4 is valid
    if b(5)<=Theta(3)
        d(6)=min(d(5)*d(1),d(2)^3);b(6)=d(6)^(1/6);  
        if b(6)<=Theta(3), m=4; return; end %Considering only two terms
        d(6)=norm1p(pB{2},3);b(6)=d(6)^(1/6);
        if b(6)<=Theta(3), m=4; return; end;
    end
    m=8; return;
end
d(8)=norm1p(pB{2},4);b(8)=d(8)^(1/8);
d(9)=d(8)*d(1);b(9)=d(9)^(1/9);
if b(8)<=Theta(4)
    d(5)=norm1pp(pB{2},2,pB{1});b(5)=d(5)^(1/5); %Test if previous m = 4 is valid
    if b(5)<=Theta(3)
        d(6)=min(d(5)*d(1),d(2)^3);b(6)=d(6)^(1/6);
        if b(6)<=Theta(3), m=4; return; end %Considering only two terms
        d(6)=norm1p(pB{2},3);b(6)=d(6)^(1/6);
        if b(6)<=Theta(3), m=4; return; end;
    end
    if b(9)<=Theta(4), m=8; return; end; %Considering only two terms
    d(9)=norm1pp(pB{2},4,pB{1});b(9)=d(9)^(1/9);
    if b(9)<=Theta(4), m=8; return; end;
end  
pB{3}=pB{2}*pB{1}; nprods=3; q=3; d(3)=norm(pB{3},1); b(3)=d(3)^(1/3);
if b(3)==0, m=2; q=2; return; end %Nilpotent matrix
d(9)=min([d(9),d(3)^3,d(2)^3*d(3)]);
d(4)=min(d(2)^2,d(3)*d(1));
d(12)=min(d(8)*d(4),d(9)*d(3));b(12)=d(12)^(1/12);
d(13)=min(d(8)*d(2)*d(3),d(9)*d(4));b(13)=d(13)^(1/13);
beta_min=max(b(12),b(13));
if beta_min<=Theta(5), m=12; return; end

%Orders for scaling
d(12)=norm1p(pB{3},4);b(12)=d(12)^(1/12);
s12aux1=max(0,ceil(log2(b(12)/Theta(5))/2)); %Tempative scaling for m = 12
d(13)=min(d(13),d(12)*d(1));b(12)=d(12)^(1/12);
s12aux2=max(0,ceil(log2(b(13)/Theta(5))/2)); %Tempative scaling for m = 12
if s12aux2>s12aux1
    d(13)=norm1pp(pB{3},4,pB{1});b(13)=d(13)^(1/13); %Only computed if it may reduce the scaling
    s12aux2=max(0,ceil(log2(b(13)/Theta(5))/2)); %Tempative scaling for m = 12
end
s12=max(s12aux1,s12aux2);
%beta_min12=max(b(12:13)); %Considering only two terms
if s12==0, m=12; return; end
%s12=ceil(log2(beta_min12/Theta(5))/2);
d(16)=min([d(12)*d(4),d(13)*d(3),d(8)^2]);b(16)=d(16)^(1/16);
d(17)=min([d(12)*d(3)*d(2),d(13)*d(4),d(16)*d(1)]);b(17)=d(17)^(1/17);
beta_min15 = max(b(16:17));
s15=max(0,ceil(log2(beta_min15/Theta(6))/2));
if s12<=s15
    d(16)=min(d(16),norm1pp(pB{3},5,pB{1}));b(16)=d(16)^(1/16);
    s15aux=max(0,ceil(log2(b(16)/Theta(6))/2)); %Temptative scaling for m=15
    if s15aux<s12    
        d(17)=min(d(17),d(16)*b(1));b(17)=d(17)^(1/17);
        beta_min15=min(beta_min15,max(b(16:17)));
        s15=max(0,ceil(log2(beta_min15/Theta(6))/2));
        if s15<s12, m=15; s=s15; return; end  %If m=15 has a lower scaling s15 and the same or less cost than m=12, m=15 is preferred
        d(17)=norm1pp(pB{3},5,pB{2});b(17)=d(17)^(1/17);
        beta_min15=min(beta_min15,max(b(16:17)));
        s15=max(0,ceil(log2(beta_min15/Theta(6))/2));
        if s15<s12, m=15; s=s15; return; else m=12; s=s12; return; end %If m=15 has a lower scaling s15 and the same or less cost than m=12, m=15 is preferred
    else m=12; s=s12; return; end %If m=12 has less cost than m=15, m=12 is preferred
else m=15; s=s15; return; end
error('cosmpol:NoParam','Cannot find valid parameters, check matrix')
end

function [m,s,q,pB,nprods]=ms_selectNoNormEst(A)
% m: 1 2 4 8 12 15
% q: 1 2 2 2  3  3
% Theta from forward and backward error bounds
Theta=[5.161913593731081e-08      %m=1  forward bound
       4.307691256676447e-05      %m=2  forward bound
       1.319680929892753e-02      %m=4  forward bound
       0.93933580333673539   %m=8  relative backward bound
       6.7523490074059511    %m=12 relative backward bound
       16.451238315562541];  %m=15 absolute forward bound Backward:9.9323454917022698 (1500 term, 150 vpa)


d=zeros(1,17); %||B^k||, B=A^2
b=zeros(1,17); %||B^k||^(1/k), B=A^2
%Initial scaling parameter s = 0 
s = 0;
% m = 1
pB{1}=A*A; nprods=1; q=1; d(1)=norm(pB{1},1); b(1)=d(1);
if b(1)<=Theta(1), m=1; return; end
% m = 2
pB{2}=pB{1}^2; nprods=2; q=2; d(2)=norm(pB{2},1); b(2)=d(2)^(1/2);
if b(2)==0, m=1; q=1; return; end %Nilpotent matrix
beta_min = beta_NoNormEst(b,d,2,q);
if beta_min<=Theta(2), m=2; return; end
% m = 4
beta_min = min(beta_min,beta_NoNormEst(b,d,4,q));
if beta_min<=Theta(3), m=4; return; end
% m = 8
beta_min = min(beta_min,beta_NoNormEst(b,d,8,q));
if beta_min<=Theta(4), m=8; return; end
% m = 12
pB{3}=pB{2}*pB{1}; nprods=3; q=3; d(3)=norm(pB{3},1); b(3)=d(3)^(1/3);
if b(3)==0, m=2; q=2; return; end %Nilpotent matrix
beta_min = min(beta_min,beta_NoNormEst(b,d,12,q));
if beta_min<=Theta(5), m=12; return; end
s12 = ceil(log2(beta_min/Theta(5))/2);
beta_min = min(beta_min,beta_NoNormEst(b,d,15,q));
s15 = max(0,ceil(log2(beta_min/Theta(6))/2)); %Scaling s=0 included
%m=12 only used for scaling if cost is lower than cost with m=15
if s12<=s15, m=12; s=s12; return; else, m=15; s=s15; return; end 
error('cosmpol:NoParam','Cannot find valid parameters, check matrix')
end

%%Auxiliary functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Beta_min = beta_NoNormEst(b,d,m,q)
% Beta_min = beta_NoNormEst(b,d,m,q) 
% Computes beta_min with only products of norms of known matrix powers
% m = order of approximation (2,4,8,12,15)
% d = norm(B^k,1), k=1:q
% b = norm(B^k,1)^1/k, k=1:q
switch m
    case 2                                             %m = 2  (m+1,m+2)
        Beta_min = (d(2)*d(1))^(1/3);
    case 4                                             %m = 4  (m+1,m+2)
        Beta_min = (d(2)^2*d(1))^(1/5);
    case 8                                             %m = 8  (m,m+1)     
        Beta_min = min(d(2)^4*d(1))^(1/9);
    case 12                                            %m = 12
        if b(2) <= b(3)                                %(m,m+1)
            Beta_min = (d(2)^5*d(3))^(1/13);
        else                                           %(m,m+1,m+2)
            Beta_min = max(min(d(3)^4*d(1),d(3)^3*d(2)^2)^(1/13),(d(3)^4*d(2))^(1/14));
        end     
    case 15                                              %m = 15
        if b(2) <= b(3)                                  %(m,m+1)
            Beta_min = (d(2)^6*d(3))^(1/15);
        else                                             %(m,m+1,m+2)
            Beta_min = max((d(3)^4*min(d(3)*d(1),d(2)^2))^(1/16),(d(3)^5*d(2))^(1/17));
        end
end
if isnan(Beta_min)||isinf(Beta_min)
    error('cosmpol:NanInfEstimation',['Nan or inf appeared in bounding the norms of high matrix powers with products of the norms of known matrix powers, check matrix'])
end
end   

function [y1, np] = cos_POL(Bp,m,q)
%   Taylor with new matrix polynomial evaluation.
%   Computes the cosine of a matrix A by means of Taylor series.
%
%   Inputs:
%      Bp: cell array with the powers of B=A^2 nedeed to compute Taylor 
%          series, such that Bp(i) contains B^i, for i=1,2,3,...,q.
%      m:  the order of approximation to be used.
%      q:  highest power of B in Bp.
%
%   Outputs: 
%      F:  the truncated Taylor series of cosine of matrix A.
%      np: number of matrix products carried out by the function.  
%
%   Revised version 1.0, Oct 6, 2015

% p=[-2, 24, -720, 40320, -3628800, 479001600, -87178291200, 20922789888000, -6402373705728000, 2432902008176640000, -1124000727777607680000, 620448401733239439360000, -403291461126605635584000000, 304888344611713860501504000000, -265252859812191058636308480000000, 263130836933693530167218012160000000, -295232799039604140847618609643520000000, 371993326789901217467999448150835200000000, -523022617466601111760007224100074291200000000, 815915283247897734345611269596115894272000000000, -1405006117752879898543142606244511569936384000000000, 2658271574788448768043625811014615890319638528000000000, -5502622159812088949850305428800254892961651752960000000000, 12413915592536072670862289047373375038521486354677760000000000, -30414093201713378043612608166064768844377641568960512000000000000, 80658175170943878571660636856403766975289505440883277824000000000000, -230843697339241380472092742683027581083278564571807941132288000000000000, 710998587804863451854045647463724949736497978881168458687447040000000000000, -2350561331282878571829474910515074683828862318181142924420699914240000000000000, 8320987112741390144276341183223364380754172606361245952449277696409600000000000000, -31469973260387937525653122354950764088012280797258232192163168247821107200000000000000, 126886932185884164103433389335161480802865516174545192198801894375214704230400000000000000, -544344939077443064003729240247842752644293064388798874532860126869671081148416000000000000000, 2480035542436830599600990418569171581047399201355367672371710738018221445712183296000000000000000, -11978571669969891796072783721689098736458938142546425857555362864628009582789845319680000000000000000, 61234458376886086861524070385274672740778091784697328983823014963978384987221689274204160000000000000000, -330788544151938641225953028221253782145683251820934971170611926835411235700971565459250872320000000000000000, 1885494701666050254987932260861146558230394535379329335672487982961844043495537923117729972224000000000000000000, -11324281178206297831457521158732046228731749579488251990048962825668835325234200766245086213177344000000000000000000, 71569457046263802294811533723186532165584657342365752577109445058227039255480148842668944867280814080000000000000000000, -475364333701284174842138206989404946643813294067993328617160934076743994734899148613007131808479167119360000000000000000000, 3314240134565353266999387579130131288000666286242049487118846032383059131291716864129885722968716753156177920000000000000000000];
% np = 0;
% I = eye(length(Bp{1}));
% F=Bp{q}/p(m);
% for jq=0:q:m-2*q
%     for i=1:(q-1)
%         F=F+Bp{q-i}/p(m-jq-i);
%     end
%     F=(F+I/p(m-jq-q))*Bp{q};
%     np=np+1;
% end
% for i=1:(q-1)
%     F=F+Bp{q-i}/p(q-i);
% end
% F=F+I;

n=size(Bp{1},1);
switch m
    case 0
        y1 = eye(n);
        np = 0;
    case 1
        y1 = -Bp{1}/2 + eye(n);
        np = 0;
    case 2
        y1 = Bp{2}/24 - Bp{1}/2 + eye(n);
        np = 0;        
    case 4
        y1 = Bp{2}*(Bp{2}/40320 - Bp{1}/720) + Bp{2}/24 - Bp{1}/2 + eye(n);
        np = 1;                
    case 8
        a= [2.186201576339059e-07
            -2.623441891606870e-05
            6.257028774393310e-03
            -4.923675742167775e-01
            1.441694411274536e-04
            5.023570505224926e+01
            1/24
            -1/2
            1];
        y0=Bp{2} * (a(1)*Bp{2}+a(2)*Bp{1});
        y1=(y0+a(3)*Bp{2}+a(4)*Bp{1}) * (y0+a(5)*Bp{2})+a(6)*y0+Bp{2}/24-Bp{1}/2+eye(n);
        np = 2;
    case 12
        a=[  1.269542268337734e-12
            -3.503936660612145e-10
            1.135275478038335e-07
            -2.027712316612395e-05
            1.647243380001247e-03
            -6.469859264308602e-01
            -4.008589447357360e-05
            9.187724869020796e-03
            -1.432942184841715e+02
            4.555439797286385e-03
            1/24
            -1/2
            1];
        y0=Bp{3} * (a(1)*Bp{3}+a(2)*Bp{2}+a(3)*Bp{1});
        y1=(y0+a(4)*Bp{3}+a(5)*Bp{2}+a(6)*Bp{1}) * (y0+a(7)*Bp{3}+a(8)*Bp{2})+a(9)*y0+a(10)*Bp{3}+Bp{2}/24-Bp{1}/2+eye(n);
        np = 2;        
    case 15
        a=[ 6.140022498994532e-17
            -2.670909787062621e-14
            1.438284920333222e-11
            -1.050202496489896e-08
            4.215975785860907e-06
            -1.238347173261219e-03
            -3.234597615453461e-09
            9.292820886910254e-07
            2.466381973203188e-01
            -9.369018510939971e-10
            1/3628800
            -1/40320
            1/720];
        y0=Bp{3} * (a(1)*Bp{3}+a(2)*Bp{2}+a(3)*Bp{1});
        y1=-((y0+a(4)*Bp{3}+a(5)*Bp{2}+a(6)*Bp{1}) * (y0+a(7)*Bp{3}+a(8)*Bp{2})+a(9)*y0+a(10)*Bp{3}+Bp{2}/3628800-Bp{1}/40320+eye(n)/720)*Bp{3}+Bp{2}/24-Bp{1}/2+eye(n);
        np = 3;        
end
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
%      gama: estimated 1-norm of A^p*B.
%
%   References: 
%   [1] FORTRAN Codes for Estimating the One-Norm of a Real or Complex
%       Matrix, with Applications to Condition Estimation.
%       ACM Transactions on Mathematical Software, Vol. 14, No. 4, 
%       pp. 381-396, 1988 (Algorithm 4.1).
%
%   [2] norm1pp function.
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
if isnan(gama)||isinf(gama)
    error('cosmpol:NanInfEstimation','Nan or inf appeared in the norm estimation of high matrix powers, check matrix or try cosmpol(A,0) (without estimation)')
end
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
if isnan(gama)||isinf(gama)
    error('cosmpol:NanInfEstimation','Nan or inf appeared in the norm estimation of high matrix powers, check matrix or try cosmpol(A,0) (without estimation)')
end
end