function [F,m,s,Cp] = expmpol(A, kmax, NormEst)
%EXPMPOL  Matrix exponential function.
%
%   [F,m,s,Cp] = expmpol(A, kmax, NormEst) computes the matrix exponential of A 
%   using a scaling and squaring algorithm, Taylor approximation and new matrix
%   polynomial evaluation methods from [2]. 
%
%   Inputs:
%      A:    the input matrix
%      kmax: a vector index to determine the maximum approximation order 
%            'm' to be used: 
%
%                                    kmax: 6  7
%            Max. approximation order (m): 24 30
%
%            If kmax<6, kmax is set to 6 (recommended value for minimum cost).
%            If kmax>7 or kmax is not provided, it is set to 7 (recommended 
%            value for a higher accuracy).
%       NormEst = 0 no estimation of norms of matrix powers is used.
%       NormEst = 1 estimation of norms of matrix powers is used.
%
%   Outputs: 
%      F:  the exponential of matrix A.
%      m:  the approximation order used.
%      s:  the scaling parameter.
%      Cp: Cost in terms of number of evaluations of matrix products.  
%
%   References:
%
%   [1] Boosting the computation of the matrix exponential, Submitted to 
%       Appl. Math. Comput., Feb. 4, 2018.
%   [2] J. Sastre, Efficient evaluation of matrix polynomials, Linear 
%       Algebra Appl., 539, 2018, 229-250.
%
%   Author: Jorge Sastre
%   Revised version: 2018/02/4.
%
%   Group of High Performance Scientific Computing (HiPerSC)
%   Universitat Politecnica de Valencia (Spain)
%   http://hipersc.blogs.upv.es

%% Check arguments
minkmax = 6;  %Minimum considered order m_M = 24 higher efficiency
maxkmax = 7;  %Maximum considered order m_M = 30 higher accuracy
if nargin<1 || nargin>3
    error('expmpol:NumParameters','Incorrect number of input arguments.');
elseif nargin==1
    kmax = maxkmax; %m=30 higher accuracy
elseif nargin<2
    if kmax<minkmax
        kmax = minkmax; %m=24 same efficiency as m=21 and higher accuracy
    elseif kmax>maxkmax
        kmax = maxkmax;
    end
end
if nargin<3
    NormEst=1;
end

% Selection of optimal order m and scaling s (Table 5 of [1])
thetam1=1.490116111983279e-8; %m=1  Forward bound
theta=[8.733457513635361e-006 %m=2  Forward bound
    1.678018844321752e-003    %m=4  Forward bound
    6.950240768069781e-02     %m=8  Forward bound
    6.925462617470704e-01     %m=15 Forward bound
    1.682715644786316         %m=21 Backward bound
    2.219048869365090         %m=24 Backward bound
    3.539666348743690];       %m=30 Backward bound

c=[4/3    %m=2   %c=abs(c_m(m+1)/c_m(m+2)) (Table 5 of [1])
   6/5    %m=4
   10/9   %m=8
   1.148757271434994 %m=15 especial calculation
   1.027657297529898 %m=21 especial calculation
   26/25   %m=24
   32/31]; %m=30

ucm2=[8.881784197001252e-016  %m=2 %ucm2=abs(u/c_m(m+2)) (Table 5 of [1])
    1.598721155460225e-014  %m=4
    4.476419235288631e-011  %m=8
    5.874311180519475e-03   %m=15
    2.935676824339517e+05   %m=21
    1.790973863109916e+09   %m=24
    9.423674633957229e+017];%m=30

if NormEst==1
     [m,s,q,maxpow,Ap,Cp] = ms_selectNormEst(A,kmax,thetam1,theta,c,ucm2);   %with norm estimation
 elseif NormEst==0
     [m,s,q,maxpow,Ap,Cp] = ms_selectNoNormEst(A,kmax,thetam1,theta,c,ucm2); %without norm estimation
end

% Scaling
if maxpow<=3 && q>3
    Ap{4}=Ap{2}^2;
    Cp=Cp+1;
    if norm(Ap{4},1)==0, m=3; q=3; s=0; end  %Nilpotent matrix
end
if maxpow<=4 && q>4
    Ap{5}=Ap{4}*A;
    Cp=Cp+1;
    if norm(Ap{5},1)==0, m=4; q=4; s=0; end  %Nilpotent matrix
end
if s
    A=A/2^s;
    for i=2:q
        Ap{i}=Ap{i}/2^(s*i);
    end
end

% Evaluation of Taylor matrix polynomial by the methods from [1, 2]
[F, np] = EFFEVPOL(A,Ap,m,q);

% Squaring
for i = 1:s 
    F = F^2;
end

% Total number of matrix products
Cp=Cp+np+s;
end

%% Selection of order m and scaling parameter s with estimation of norms of powers of A
function [m,s,q,maxpow,Ap,Cp]=ms_selectNormEst(A,kmax,thetam1,theta,c,ucm2)
% [m,s,q,maxpow,Ap,Cp]=ms_selectNoNormEst(A,kmax,thetam1,theta,c,ucm2)
% Given a square matrix A this function obtains the order m and scaling parameter s
% for Taylor approximation of order m<=m_M=24 (if kmax=6) or 30 (if kmax=7) of
% the matrix exponential using estimations of norms of matrix powers
% Optimal orders m:            2 4 8 15 21 24
% Corresponding matrix powers: 2 2 2  2  3  4
%
%   Inputs:
%      A: the input matrix.
%      kmax: index for the maximum order m to be used  m
%      m_M=24 (if kmax=6) or 30 (if kmax=7)
%      thetam1, theta, c, ucm2 parameters from Table 5 of [1]
%
%   Outputs:
%      m: order or approximation to be used.
%      s: scaling parameter
%      q: maximum matrix power to be used in Taylor approximation (A,A^2,...,A^q)
%      maxpow: maximum computed power of matrix A (maxpow<=q)
%      Ap: cell array with the powers of A, A^2,A^3,...,A^maxpow.
%      Cp: matrix products carried out by the function. 

n = size(A,1);
%Initial scaling parameter s = 0
s = 0;
Cp = 0;
Ap=cell(5,1);
a=realmax*ones(1,26); %vector of ||A^k|| (if a(i)=realmax: not calculated)
% m = 1
a(1) = norm(A,1);
if a(1)<=thetam1, q=1; maxpow=1; m=1; return; end
% m = 2
Ap{2} = A*A; a(2)=norm(Ap{2},1); Cp=1; q=2; maxpow=2;
if a(2)==0, m=1; q=1; return; end %Nilpotent matrix
a(3)=a(1)*a(2); a(4)=a(2)^2; b(1)=max(1,a(1))*ucm2(1);
if c(1)*a(3)+a(4)<=b(1), m=2; return; end 
% m = 4
a(5)=a(4)*a(1); a(6)=a(4)*a(2); b(2)=max(1,a(1))*ucm2(2);
if c(2)*a(5)+a(6)<=b(2)
    a(3)=norm1pp(Ap{2},1,A); %Test if previous m = 2 is valid
    if c(1)*a(3)<=b(1)
        a(4)=min(a(4),a(3)*a(1));
        if c(1)*a(3)+a(4)<=b(1), m=2; return, end
        a(4)=norm1p(Ap{2},2);
        if c(1)*a(3)+a(4)<=b(1), m=2; return, end
    end
    m=4; return;
end 
% m = 8
a(9)=a(5)*a(4); a(10)=a(6)*a(4); b(3)=max(1,a(1))*ucm2(3);
if c(3)*a(9)+a(10)<=b(3)
    a(5)=norm1pp(Ap{2},2,A); %Test if previous m = 4 is valid
    if c(2)*a(5)<=b(2)
        a(6)=min(a(6),a(5)*a(1));
        if c(2)*a(5)+a(6)<=b(2), m=4; return, end
        a(6)=norm1p(Ap{2},3);
        if c(2)*a(5)+a(6)<=b(2), m=4; return, end
    end  
    m=8; return;
end 
% m = 15 no estimation
a(16)=a(2)^8; a(17)=a(16)*a(1); b(4)=max(1,a(1))*ucm2(4);
if c(4)*a(16)+a(17)<=b(4)
    a(9)=norm1pp(Ap{2},4,A); %Test if previous m = 8 is valid
    if c(3)*a(9)<=b(3)
        a(10)=min(a(10),a(9)*a(1));
        if c(3)*a(9)+a(10)<=b(3), m=8; return, end
        a(10)=norm1p(Ap{2},5);
        if c(3)*a(9)+a(10)<=b(3), m=8; return, end
    end      
    m=15; return; 
end 
% m = 15 with estimation
a(16)=norm1p(Ap{2},8); a(17)=a(16)*a(1);
if c(4)*a(16)<=b(4)
    a(9)=norm1pp(Ap{2},4,A); %Test if previous m = 8 is valid
    if c(3)*a(9)<=b(3)
        a(10)=min(a(10),a(9)*a(1));
        if c(3)*a(9)+a(10)<=b(3), m=8; return, end
        a(10)=norm1p(Ap{2},5);
        if c(3)*a(9)+a(10)<=b(3), m=8; return, end
    end
    if c(4)*a(16)+a(17)<=b(4), m=15; return; end
    a(17) = norm1pp(Ap{2},8,A);
    if c(4)*a(16)+a(17)<=b(4), m=15; return; end
end

% m = 21
Ap{3} = Ap{2}*A; a(3)=norm(Ap{3},1); Cp=2; q=3; maxpow=3;
if a(3)==0, m=2; q=2; return; end %Nilpotent matrix
a(4)=min(a(4),a(3)*a(1));a(5)=min(a(5),a(2)*a(3));a(6)=min(a(6),a(3)^2);a(7)=min(a(5)*a(2),a(6)*a(1));
a(22)=min([a(16)*a(6),a(17)*a(5),a(2)^11,a(3)^6*a(2)^2,a(3)^7*a(1)]); 
a(23)=min([a(16)*a(7),a(17)*a(6),a(2)^10*a(3),a(3)^7*a(2)]);
b(5)=max(1,a(1))*ucm2(5);
if c(5)*a(22)+a(23)<=b(5), m=21; return; end

% m = 24
a(8)=a(6)*a(2);a(9)=min(a(9),a(3)^3);a(10)=min(a(10),a(9)*a(1));
a(25)=min([a(16)*a(9),a(17)*a(8),a(2)^11*a(3),a(3)^7*a(2)^2,a(3)^8*a(1)]);
a(26)=min([a(16)*a(10),a(17)*a(9),a(2)^13,a(3)^8*a(2)]);
b(6)=max(1,a(1))*ucm2(6);
if c(6)*a(25)+a(26)<=b(6)
    a(22)=norm1pp(Ap{3},7,A); %Test if previous m = 21 is valid
    if c(5)*a(22)<=b(5)
        a(23)=min(a(23),a(22)*a(1));
        if c(5)*a(22)+a(23)<=b(5), m=21; return, end
        a(23)=norm1pp(Ap{3},7,Ap{2});
        if c(5)*a(22)+a(23)<=b(5), m=21; return, end
    end
    m=24; q=4; return;
end
%Orders for scaling m=21 and mmax=24
if kmax==6 
    a(25)=min(a(25),norm1pp(Ap{3},8,A));a(26)=min(a(26),a(25)*a(1));
    a26estimated=false;
    if c(6)*a(25)<=b(6)
        a(22)=norm1pp(Ap{3},7,A); %Test if previous m = 21 is valid
        if c(5)*a(22)<=b(5)
            a(23)=min(a(23),a(22)*a(1));
            if c(5)*a(22)+a(23)<=b(5), m=21; return, end
            a(23)=norm1pp(Ap{3},7,Ap{2});
            if c(5)*a(22)+a(23)<=b(5), m=21; return, end
        end
        if c(6)*a(25)+a(26)<=b(6), m=24; q=4; return, end
        a(26)=min(a(26),norm1pp(Ap{3},8,Ap{2}));
        a26estimated=true;
        if c(6)*a(25)+a(26)<=b(6), m=24; q=4; return, end
    end
    if ~a26estimated
        a(26)=min(a(26),norm1pp(Ap{3},8,Ap{2}));
    end    
    % Compute alpha_min for m=24 inm=6
    alpha_min = max(a(25)^(1/25), a(26)^(1/26));
    [t s] = log2(alpha_min/theta(6));
    s = s - (t == 0.5); % adjust s if alpha_min/theta(kmax) is a power of 2.
    % Test if s can be reduced
    if s>0
        sred = s-1;
        b(6) = max(1,a(1)/2^sred)*ucm2(6);
        if c(6)*a(25)/2^(25*sred)+a(26)/2^(26*sred)<=b(6)
            % s can be reduced
            s = sred;
        end
    end
    % Test if the scaled matrix allows using m=21 inm=5
    a(22)=norm1pp(Ap{3},7,A);
    a(23)=min(a(23),a(22)*a(1));
    b(5) = max(1,a(1)/2^s)*ucm2(5);
    if c(5)*a(22)/2^(22*s)<=b(5)
        if c(5)*a(22)/2^(22*s)+a(23)/2^(23*s)<=b(5), m=21; return, end  % The scaled matrix allows using 21
        a(23)=norm1pp(Ap{3},7,Ap{2});
        if c(5)*a(22)/2^(22*s)+a(23)/2^(23*s)<=b(5), m=21; return, end  % The scaled matrix allows using 21
    end
    m=24; q=4; return
%Orders for scaling m=24 and mmax=30
else
% m = 24
    Ap{4} = Ap{2}^2; a(4)=norm(Ap{4},1); Cp=3; q=4; maxpow=4;
    if a(4)==0, m=3; q=3; return; end %Nilpotent matrix
    a(25)=min([a(25),a(16)*a(4)^2*a(1),a(17)*a(4)^2,a(3)^7*a(4)]);
    a(26)=min([a(26),a(16)*a(4)^2*a(2),a(17)*a(4)^2*a(1),a(3)^7*a(4)*a(1)]);
    b(6)=max(1,a(1))*ucm2(6);
    if c(6)*a(25)+a(26)<=b(6), m=24; return; end
% m = 30
    a(14)=min(a(4)^3,a(3)^4)*a(2);a(15)=min(a(4)^3*a(3),a(3)^5);
    a(31)=min([a(16)*a(15),a(17)*a(14),a(3)^9*a(4)]);
    a(32)=min([a(16)^2,a(17)*a(15),a(3)^10*a(2),a(31)*a(1)]);
    b(7)=max(1,a(1))*ucm2(7);
    if c(7)*a(31)+a(32)<=b(7)
        a(25)=norm1pp(Ap{4},6,A); %Test if previous m = 24 is valid
        if c(6)*a(25)<=b(6)
            a(26)=min(a(26),a(25)*a(1));
            if c(6)*a(25)+a(26)<=b(6), m=24; return, end
            a(26)=norm1pp(Ap{4},6,Ap{2});
            if c(6)*a(25)+a(26)<=b(6), m=24; return, end
        end
        m=30; q=5; return;
    end
    a(31)=min(a(31),norm1pp(Ap{4},7,Ap{3}));a(32)=min(a(32),a(31)*a(1));
    a32estimated=false;
    if c(7)*a(31)<=b(7)
        a(25)=norm1pp(Ap{4},6,A); %Test if previous m = 24 is valid
        if c(6)*a(25)<=b(6)
            a(26)=min(a(26),a(25)*a(1));
            if c(6)*a(25)+a(26)<=b(6), m=24; return, end
            a(26)=norm1pp(Ap{4},6,Ap{2});
            if c(6)*a(25)+a(26)<=b(6), m=24; return, end
        end
        if c(7)*a(31)+a(32)<=b(7), m=30; q=5; return, end
        a(32)=min(a(32),norm1p(Ap{4},8));
        a32estimated=true;
        if c(7)*a(31)+a(32)<=b(7), m=30; q=5; return, end
    end
    if ~a32estimated
        a(32)=min(a(32),norm1p(Ap{4},8));
    end
    % Compute alpha_min for m=30 inm=7
    alpha_min = max(a(31)^(1/31), a(32)^(1/32));
    [t s] = log2(alpha_min/theta(7));
    s = s - (t == 0.5); % adjust s if normA/theta(kmax) is a power of 2.
    % Test if s can be reduced
    if s>0
        sred = s-1;
        b(7) = max(1,a(1)/2^sred)*ucm2(7);
        if c(7)*a(31)/2^(31*sred)+a(32)/2^(32*sred)<=b(7)
            % s can be reduced
            s = sred;
        end
    end
    % Test if the scaled matrix allows using m=24 inm=6
    a(25)=norm1pp(Ap{4},6,A);
    a(26)=min(a(26),a(25)*a(1));
    b(6) = max(1,a(1)/2^s)*ucm2(6);
    if c(6)*a(25)/2^(25*s)<=b(6)
        if c(6)*a(25)/2^(25*s)+a(26)/2^(26*s)<=b(6), m=24; return, end  % The scaled matrix allows using 24
        a(26)=norm1pp(Ap{4},6,Ap{2});
        if c(6)*a(25)/2^(25*s)+a(26)/2^(26*s)<=b(6), m=24; return, end  % The scaled matrix allows using 24
    end
    Ap{5} = Ap{4}*A; a(5)=norm(Ap{5},1); Cp=4; q=5; maxpow=5;
    if a(5)==0, m=4; q=4; s=0; return; end %Nilpotent matrix
    m=30; q=5; return
end
end

%% Selection of order m and scaling parameter s with no estimation of norms of powers of A
function [m,s,q,maxpow,Ap,Cp]=ms_selectNoNormEst(A,kmax,thetam1,theta,c,ucm2)
% [m,s,q,maxpow,Ap,Cp]=ms_selectNoNormEst(A,kmax,thetam1,theta,c,ucm2)
% Given a square matrix A this function obtains the order m and scaling parameter s
% for Taylor approximation of order m<=m_M=24 (if kmax=6) or 30 (if kmax=7) of
% the matrix exponential using no estimations of norms of matrix powers
% Optimal orders m:            2 4 8 15 21 24
% Corresponding matrix powers: 2 2 2  2  3  4
%
%   Inputs:
%      A: the input matrix.
%      kmax: index for the maximum order m to be used  m
%      m_M=24 (if kmax=6) or 30 (if kmax=7)
%      thetam1, theta, c, ucm2 parameters from Table 5 of [1]
%
%   Outputs:
%      m: order or approximation to be used.
%      s: scaling parameter
%      q: maximum matrix power to be used in Taylor approximation (A,A^2,...,A^q)
%      maxpow: maximum computed power of matrix A (maxpow<=q)
%      Ap: cell array with the powers of A, A^2,A^3,...,A^maxpow.
%      Cp: matrix products carried out by the function. 

n = size(A,1);
%Initial scaling parameter s = 0
s = 0;
Cp = 0;
Ap=cell(5,1);
a=realmax*ones(1,26); %vector of ||A^k|| (if a(i)=realmax: not calculated)
% m = 1
a(1) = norm(A,1);
if a(1)<=thetam1, q=1; maxpow=1; m=1; return; end
% m = 2
Ap{2} = A*A; a(2)=norm(Ap{2},1); Cp=1; q=2; maxpow=2;
if a(2)==0, m=1; q=1; return; end %Nilpotent matrix
a(3)=a(1)*a(2); a(4)=a(2)^2; b=max(1,a(1))*ucm2(1);
if c(1)*a(3)+a(4)<=b, m=2; return; end 
% m = 4
a(5)=a(4)*a(1); a(6)=a(4)*a(2); b=max(1,a(1))*ucm2(2);
if c(2)*a(5)+a(6)<=b, m=4; return; end 
% m = 8
a(9)=a(5)*a(4); a(10)=a(6)*a(4); b=max(1,a(1))*ucm2(3);
if c(3)*a(9)+a(10)<=b, m=8; return; end 
% m = 15
a(16)=a(2)^8; a(17)=a(16)*a(1); b=max(1,a(1))*ucm2(4);
if c(4)*a(16)+a(17)<=b, m=15; return; end 
% m = 21
Ap{3} = Ap{2}*A; a(3)=norm(Ap{3},1); Cp=2; q=3; maxpow=3;
if a(3)==0, m=2; q=2; return; end %Nilpotent matrix
a(22)=min([a(2)^11,a(3)^6*a(2)^2,a(3)^7*a(1)]); 
a(23)=min([a(2)^10*a(3),a(3)^7*a(2)]);
b=max(1,a(1))*ucm2(5);
if c(5)*a(22)+a(23)<=b, m=21; return; end 
if kmax==6 %mmax=24
    % m = 24
    a(25)=min([a(2)^11*a(3),a(3)^7*a(2)^2,a(3)^8*a(1)]);
    a(26)=min([a(2)^13,a(3)^8*a(2)]);
    b=max(1,a(1))*ucm2(6);
    if c(6)*a(25)+a(26)<=b, m=24; q=4; return; end

    % Compute alpha_min for m=24 inm=6
    alpha_min = max(a(25)^(1/25), a(26)^(1/26));
    [t s] = log2(alpha_min/theta(6));
    s = s - (t == 0.5); % adjust s if normA/theta(kmax) is a power of 2.
    
    % Test if s can be reduced
    if s>0
        sred = s-1;
        b = max(1,a(1)/2^sred)*ucm2(6);
        if c(6)*a(25)/2^(25*sred)+a(26)/2^(26*sred)<=b
            % s can be reduced
            s = sred;
        end
    end
    
    % Test if the scaled matrix allows using m=21 inm=5
    b = max(1,a(1)/2^s)*ucm2(5);
    if c(5)*a(22)/2^(22*s)+a(23)/2^(23*s)<=b
        m = 21;  % The scaled matrix allows using 21
    else
        q = 4;
        m = 24;
    end
else %mmax=30 kmax=7
% m = 24
    Ap{4} = Ap{2}^2; a(4)=norm(Ap{4},1); Cp=3; q=4; maxpow=4;
    if a(4)==0, m=3; q=3; return; end %Nilpotent matrix
    a(25)=min([a(4)^6*a(1),a(3)^7*a(4)]);
    a(26)=min([a(4)^6,a(3)^8])*a(2);
    b=max(1,a(1))*ucm2(6);
    if c(6)*a(25)+a(26)<=b, m=24; return; end
% m = 30
    a(31)=min([a(4)^7*a(3),a(3)^9*a(4)]);
    a(32)=min([a(4)^8,a(3)^10*a(2),a(31)*a(1)]);
    b=max(1,a(1))*ucm2(7);
    if c(7)*a(31)+a(32)<=b, m=30; q=5; return; end   
    
    % Compute alpha_min for m=24 inm=6
    alpha_min = max(a(31)^(1/31), a(32)^(1/32));
    [t s] = log2(alpha_min/theta(7));
    s = s - (t == 0.5); % adjust s if normA/theta(kmax) is a power of 2.
    
    % Test if s can be reduced
    if s>0
        sred = s-1;
        b = max(1,a(1)/2^sred)*ucm2(7);
        if c(7)*a(31)/2^(31*sred)+a(32)/2^(32*sred)<=b
            % s can be reduced
            s = sred;
        end
    end
    
    % Test if the scaled matrix allows using m=24 inm=6
    b = max(1,a(1)/2^s)*ucm2(6);
    if c(6)*a(25)/2^(25*s)+a(26)/2^(26*s)<=b
        m = 24;  % The scaled matrix allows using 24
    else
        Ap{5} = Ap{4}*A; a(5)=norm(Ap{5},1); Cp=4; q=5; maxpow=5;
        if a(5)==0, m=4; q=4; s=0; return; end %Nilpotent matrix
        m = 30;
    end    
end
end

%% Taylor approximations evaluation with the methods from [2]
function [sol, np] = EFFEVPOL(A,Ap,m,q)
%   [sol, np] = EFFEVPOL(A,Ap,m,q)
%   Computes the exponential of matrix A by means of Taylor series
%   and matrix polynomial evaluation methods from [1, 2].
%
%   Inputs:
%      A:  the input matrix.
%      Ap: cell array with the powers of A nedeed to compute Taylor
%          series, such that Ap{i} contains A^i, for i=2,3,...,q.
%      m:  order or approximation to be used.
%      q:  maximum matrix power used (A,A^2,...,A^q)
%
%   Outputs:
%      sol:  the exponential of matrix A.
%      np: matrix products carried out by the function.
%
%   Revised version 2018/01/10
n = size(A);
switch m
    case 1 %m=1 q=1
        sol=A+eye(n);
        np = 0;
    case 2 %m=2 q=2
        sol=Ap{2}/2+A+eye(n);
        np = 0;
    case 3 %m=3 q=3 %Nilpotent matrices
        sol=Ap{3}/6+Ap{2}/2+A+eye(n);
        np = 0;        
    case 4 %m=4 
        if q==2
            sol=((Ap{2}/4+A)/3+eye(n))*Ap{2}/2+A+eye(n);
            np=1;
        else %q=4 Nilpotent matrices
            sol=((Ap{4}/4+Ap{3})/3+Ap{2})/2+A+eye(n);
            np=0;
        end
    case 8 %m=8 q=2
        c=[4.980119205559973e-03     1.992047682223989e-02     7.665265321119147e-02     8.765009801785554e-01    1.225521150112075e-01     2.974307204847627 0.5 1 1];
        y0s=Ap{2}*(c(1)*Ap{2}+c(2)*A);
        sol=(y0s+c(3)*Ap{2}+c(4)*A)*(y0s+c(5)*Ap{2})+c(6)*y0s+Ap{2}/2+A+eye(n);
        np=2;
    case 15 %m=15 q=2
        c=[4.018761610201036e-04     2.945531440279683e-03    -8.709066576837676e-03     4.017568440673568e-01     3.230762888122312e-02     5.768988513026145e+00 2.338576034271299e-02     2.381070373870987e-01     2.224209172496374e+00    -5.792361707073261e+00    -4.130276365929783e-02     1.040801735231354e+01 -6.331712455883370e+01     3.484665863364574e-01 1 1];
        y0s=Ap{2}*(c(1)*Ap{2}+c(2)*A);
        y1s=(y0s+c(3)*Ap{2}+c(4)*A)*(y0s+c(5)*Ap{2})+c(6)*y0s+c(7)*Ap{2};
        sol=(y1s+c(8)*Ap{2}+c(9)*A)*(y1s+c(10)*y0s+c(11)*A)+c(12)*y1s+c(13)*y0s+c(14)*Ap{2}+A+eye(n);
        np=3;
    case 21 %m=21 q=3
        c=[1.161658834444880e-06     4.500852739573010e-06     5.374708803114821e-05     2.005403977292901e-03     6.974348269544424e-02     9.418613214806352e-01 2.852960512714315e-03    -7.544837153586671e-03     1.829773504500424e+00     3.151382711608315e-02     1.392249143769798e-01    -2.269101241269351e-03 -5.394098846866402e-02     3.112216227982407e-01     9.343851261938047e+00     6.865706355662834e-01     3.233370163085380e+00    -5.726379787260966e+00 -1.413550099309667e-02    -1.638413114712016e-01 1 1];
        y0s=Ap{3}*(c(1)*Ap{3}+c(2)*Ap{2}+c(3)*A);
        y1s=(y0s+c(4)*Ap{3}+c(5)*Ap{2}+c(6)*A)*(y0s+c(7)*Ap{3}+c(8)*Ap{2})+c(9)*y0s+c(10)*Ap{3}+c(11)*Ap{2};
        sol=(y1s+c(12)*Ap{3}+c(13)*Ap{2}+c(14)*A)*(y1s+c(15)*y0s+c(16)*A)+c(17)*y1s+c(18)*y0s+c(19)*Ap{3}+c(20)*Ap{2}+A+eye(n);
        np=3;
    case 24 %m=24 q=4
        c=[1.172460202011541e-08     9.379681616092325e-08     1.406952242413849e-06     2.294895435403922e-05     2.024281516007681e-03     1.430688980356062e-02     1.952545843107103e-01     2.865001388641538e+00    -1.204349003694297e-03     2.547056607231984e-03 2.721930992200371e-02     2.498969092549990e+02     2.018492049443954e-02     1.965098904519709e-01     1.739158441630994e+00 8.290085751394409e+00     2.919349464582001e-04     1.758035313846159e-04     1.606091400855144e-02     3.655234395347475e-02 2.243394407902074e-03    -3.005000525808178e-02     1.969779342112314e-01     1     1];
        y0s=Ap{4}*(c(1)*Ap{4}+c(2)*Ap{3}+c(3)*Ap{2}+c(4)*A);
        y1s=(y0s+c(5)*Ap{4}+c(6)*Ap{3}+c(7)*Ap{2}+c(8)*A)*(y0s+c(9)*Ap{4}+c(10)*Ap{3}+c(11)*Ap{2})+c(12)*y0s+c(13)*Ap{4}+c(14)*Ap{3}+c(15)*Ap{2}+c(16)*A;
        sol=y1s*(y0s+c(17)*Ap{4}+c(18)*Ap{3}+c(19)*Ap{2}+c(20)*A)+c(21)*Ap{4}+c(22)*Ap{3}+c(23)*Ap{2}+A+eye(n);
        np=3;
    case 30 %m=30 q=5
        c=[1.556371639324141e-11     1.556371639324141e-10     2.957106114715868e-09     6.204734935438909e-08     1.313681421698863e-06     3.501669195497238e-05 1.283057135586989e-03     2.479095151834799e-02     4.155284057336423e-01     5.951585263506065e+00     3.753710741641900e-05     2.100333647757715e-04 2.630043177655382e-03     3.306559506631931e-02     6.175954247606858e+01     2.742336655922557e-03     3.005135891320298e-02     2.857950268422422e-01 2.991654767354374e+00     1.110689398085882e+01     8.572383602707347e-06     9.027588625491207e-05     1.121744731945438e-03     8.139086096860678e-03 -2.638236222337760e-04     6.263526066651383e-05     4.985549176118462e-03     7.705596948494946e-02     5.029302610017967e-01 1 1];
        y0s=Ap{5}*(c(1)*Ap{5}+c(2)*Ap{4}+c(3)*Ap{3}+c(4)*Ap{2}+c(5)*A);
        y1s=(y0s+c(6)*Ap{5}+c(7)*Ap{4}+c(8)*Ap{3}+c(9)*Ap{2}+c(10)*A)*(y0s+c(11)*Ap{5}+c(12)*Ap{4}+c(13)*Ap{3}+c(14)*Ap{2})+c(15)*y0s+c(16)*Ap{5}+c(17)*Ap{4}+c(18)*Ap{3}+c(19)*Ap{2}+c(20)*A;
        sol=y1s*(y0s+c(21)*Ap{5}+c(22)*Ap{4}+c(23)*Ap{3}+c(24)*Ap{2}+c(25)*A)+c(26)*Ap{5}+c(27)*Ap{4}+c(28)*Ap{3}+c(29)*Ap{2}+A+eye(n);
        np=3;
end
end

%% Estimation of norms of matrix powers
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

