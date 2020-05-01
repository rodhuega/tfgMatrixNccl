function [A,m,s,nprods] = cosmtayher(A)
%   COSMTAYHER computes matrix cosine by means Hermites series.
%   cosmtayher(A) computes the matrix cosine cos(A) using a scaling 
%   and squaring algorithm and the Hermites approximation with maximum order equal to 32.
%   Function returns the matrix cosine, A; the appoximation
%   order used m; the scaling parameter s; and nprods in terms of 
%   number of matrix product evaluations.
%   This code (see [1]) is based on cosmtay (see [2]), where the polynomial coefficients of
%   Taylor series of the cosine function have been replaced by the
%   coefficients of Hermite series of the cosine function. Also, the Theta
%   values have been modified.
%   
%
%   References:
%   [1] E. Defez, J. Ibáñez, J. Peinado, J. Sastre, P. Alonso. An efficient 
%       and accurate algorithm for computing the matrix cosine based on New
%       Hermite approximations, submitted to Applied Mathematics and Computation, 
%       January 2018.
%   [2] J. Sastre, J. Ibáñez, P. Alonso, J. Peinado, E. Defez. 
%       Two sequential algorithms for computing the matrix cosine function, 
%       Applied Mathematics and Computation, 312(2017), pp. 66-77. 
%
%   Authors: Jorge Sastre and J. Javier.
%   Revised version: 2018/09/18.
%
%   Group of High Performance Scientific Computation (HiPerSC)
%   Universitat Politecnica de Valencia (Spain)
%   http://hipersc.blogs.upv.es
%  

%% Check arguments
if nargin~=1
    error('cosmtayher:NumParameters','Incorrect number of input arguments.');
end
[m,s,q,Bp,nprods] = ms_selectNoNormEst(A);

% Scaling
if isinf(s)
    s=10;
end
if s
    for i=1:q
        Bp{i}=Bp{i}/4^(s*i);
    end
end
% Evaluation of Taylor matrix polynomial by Paterson-Stockmeyer method
[A,npTPS]=cos_TPS(Bp,m);
% Recovering cos(A) from the scaled aproximation cos(2^(-s)*A)
I=eye(size(A));
for i=1:s
    A=2*A*A-I;
end
% Total number of matrix products
nprods=nprods+npTPS+s;
end


function [m,s,q,pB,nprods]=ms_selectNoNormEst(A)
% m: 1 2 4 6 9 12 16
% q: 1 2 2 3 3  4  4
% Theta from forward and backward error bounds

Theta=[ 
        3.7247156392713732e-05  %m= 2
        0.011723485672548786    %m= 4
        0.17002640749360201     %m= 6
        1.623738709885105       %m= 9
        6.16270324616079        %m=12
        20.113380730824375      %m=16
      ];
d=zeros(1,17); %||B^k||, B=A^2
b=zeros(1,17); %||B^k||^(1/k), B=A^2
%Initial scaling parameter s = 0 
s = 0;
% m = 1
pB{1}=A*A; d(1)=norm(pB{1},1); b(1)=d(1);
% m = 2
pB{2}=pB{1}^2; nprods=2; q=2; d(2)=norm(pB{2},1); b(2)=d(2)^(1/2);
if b(2)==0, m=1; q=1; return; end %Nilpotent matrix
beta_min = beta_NoNormEst(b,d,2,q);
if beta_min<=Theta(1), m=2; return; end
% m = 4
beta_min = min(beta_min,beta_NoNormEst(b,d,4,q));
if beta_min<=Theta(2), m=4; return; end
% m = 6
pB{3}=pB{2}*pB{1}; nprods=3; q=3; d(3)=norm(pB{3},1); b(3)=d(3)^(1/3);
if b(3)==0, m=2; q=2; return; end %Nilpotent matrix
d(6)=min(d(3)^2,d(2)^3);
beta_min = min(beta_min,beta_NoNormEst(b,d,6,q));
if beta_min<=Theta(3), m=6; return; end
% m = 9
beta_min9 = min(beta_min,beta_NoNormEst(b,d,9,q));
if beta_min9<=Theta(4), m=9; return; end
% m = 12
beta_min12 = min(beta_min9,beta_NoNormEst(b,d,12,q));
if beta_min12<=Theta(5), m=12; return; end
%m=9 only used for scaling if cost is lower than cost with m=12,16
s9 = ceil(log2(beta_min9/Theta(4))/2); %Scaling s=0 not included
s12 = ceil(log2(beta_min12/Theta(5))/2);
if s9<=s12, m=9; s=s9; return; end
% m = 12
pB{4}=pB{3}*pB{1}; nprods=4; q=4; d(4)=norm(pB{4},1); b(4)=d(4)^(1/4);
if b(4)==0, m=3; q=3; return; end %Nilpotent matrix
d(6)=min(d(6),d(4)*d(2));
beta_min12 = min(beta_min12,beta_NoNormEst(b,d,12,q)); %We have new information with pB{4}=B^4
if beta_min12<=Theta(5), m=12; s=0; return; end
%m=12 only used for scaling if cost is lower than cost with m=16
s12 = ceil(log2(beta_min12/Theta(5))/2);
beta_min16 = min(beta_min12,beta_NoNormEst(b,d,16,q));
s16 = max(0,ceil(log2(beta_min16/Theta(6))/2)); %Scaling s=0 included
if s12<=s16
    m=12; s=s12;
    return; 
else
    m=16; s=s16;
end 
end

%%Auxiliary functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Beta_min = beta_NoNormEst(b,d,m,q)
% Beta_min = beta_NoNormEst(b,d,m,q) 
% Computes beta_min with only products of norms of known matrix powers
% m = order of approximation (2,4,6,9,12,16,>=20)
% d = norm(B^k,1), k=1:q
% b = norm(B^k,1)^1/k, k=1:q
switch m
    case 2                                             %k=1,2
        Beta_min = d(1);
    case 4                                             %k=2,3
        Beta_min = (d(2)*d(1))^(1/3);
    case 6                                             %k=3,4
        if q == 2
            Beta_min = max(d(2)*d(1))^(1/3);
        elseif q == 3
            Beta_min = max(d(3)^(1/3),d(2)^(1/2));
        end    
    case 9                                             %m = 9
        if b(2) <= b(3)
            Beta_min = (d(2)^3*d(3))^(1/9);
        else
            Beta_min = max(min(d(2)^2*d(3)^2,d(3)^3*d(1))^(1/10),(d(3)^3*d(2))^(1/11));
        end
    case 12                                              %m = 12       
         if q == 3 
            if b(2) <= b(3)
                Beta_min = (d(2)^5*d(3))^(1/13);
            else
                Beta_min = min(d(3)^4*d(1),d(3)^3*d(2)^2)^(1/13);
            end
        elseif q == 4
            if b(3) <= b(4)
                Beta_min = max((d(3)^3*d(4))^(1/13),min(d(3)^2*d(4)^2,d(3)^4*d(2))^(1/14));
            else
                Beta_min = max((d(4)^2*min(d(3)*d(2),d(4)*d(1)))^(1/13),(d(4)^2*d(6))^(1/14));
            end                    
        end       
    case 16                                              %m = 16
        if b(3) <= b(4)
            Beta_min = max((d(3)^4*d(4))^(1/16),min(d(3)^5*d(2),d(3)^3*d(4)^2)^(1/17));
        else
            Beta_min = max((d(4)^3*min(d(4)*d(1),d(3)*d(2)))^(1/17),(d(4)^3*d(6))^(1/18));
        end
    otherwise                                            %m >= 20
        if b(3) <= b(4)
            Beta_min = max(min(d(3)^6*d(2),d(3)^4*d(4)^2)^(1/20),(d(3)^6*d(4))^(1/22));
        else
            Beta_min = max(min(d(4)^5*d(1),d(4)^4*d(3)*d(2))^(1/21),(d(4)^4*d(6))^(1/22));
        end
end
if isnan(Beta_min)||isinf(Beta_min)
    error('cosmtayher:NanInfEstimation',['Nan or inf appeared in bounding the norms of high matrix powers with products of the norms of known matrix powers, check matrix'])
end
end   

function [B, np] = cos_TPS(A,m)
%   Taylor Paterson-Stockmeyer algorithm.
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
%   Revised version 2.0, Jun. 27, 2018

switch m
    case 2
		p=[	0.99999999999999995668608239024668
		-0.49999999999979884953785261674606
		0.041666632413903896446707459803832
		];
    case 4
 		p=[	1.0000000000000000033511529244153
		-0.49999999999999999821440959971895
		0.041666666666658229298944463133192
		-0.0013888888815432802664376784185645
		0.000024799100332139193413999282857613
		];
    case 6
		p=[	1.0000000000000000392941191474293
		-0.50000000000000001964671139058827
		0.041666666666666668150650137563611
		-0.0013888888888888621770088061703515
		0.000024801587299405915760381171473171
		-0.00000027557310496548087636865460679949
		0.000000002086034180841412041959113403662
		];
    case 9
		p=[	1.0000000000000000301940941523455
		-0.50000000000000001509704707439343
		0.041666666666666667924753785947679
		-0.0013888888888888889308199551758237
		0.000024801587301587302214744028149105
		-0.00000027557319223985707110381332170183
		0.0000000020876756987693483399240792350193
		-0.000000000011470745498166271952977354971349
		0.000000000000047794444565365398753592523820976
		-0.0000000000000001556163142302421150082695797044
		];
    case 12
		p=[	0.99999999999999998860416174700896
		-0.49999999999999999430208083513663
		0.041666666666666666191839177391527
		-0.0013888888888888888730539861256211
		0.000024801587301587301281507045922108
		-0.00000027557319223985887992651350715745
		0.0000000020876756987868099922378257193038
		-0.000000000011470745597729717808945110755012
		0.000000000000047794773323855347114099024987862
		-0.00000000000000015619206965130746293859277800157
		0.00000000000000000041103172029160940302168128973484
		-0.00000000000000000000088964729059837614501030539572818
		0.0000000000000000000000015981626940717157395171772101859
		];
    case 16
 		p=[	1.0000000000000000264314888211006
		-0.50000000000000001321574386692736
		0.041666666666666667767972152044951
		-0.0013888888888888889255716802906418
		0.000024801587301587302198184006173088
		-0.00000027557319223985889064053204934899
		0.0000000020876756987868100717340688229757
		-0.000000000011470745597729724811918769482031
		0.000000000000047794773323873853748216405131158
		-0.00000000000000015619206968586225700009018975678
		0.00000000000000000041103176233121647065737844653463
		-0.00000000000000000000088967913924503417971272413923669
		0.0000000000000000000000016117375710808582111437659105651
		-0.0000000000000000000000000024795962560897445143148901110448
		0.0000000000000000000000000000032798869291448378567245544078104
		-3.7694985101890663043088959879089e-33
		3.7394485486773398040581720801657e-36
		];
end
np = 0;
n=size(A{1},1);
I=eye(n);
switch m
    case 1
        B=A{1}*p(2) + I*p(1);
    case 2
         B=A{2}*p(3)+ A{1}*p(2) + I*p(1);
	case 4
        B=A{2}*p(5) + A{1}*p(4) + I*p(3);
        B=B*A{2}    + A{1}*p(2) + I*p(1);
        np=1;
	case 6 
        B=A{2}*p(7) + A{1}*p(6) + I*p(5);
        B=B*A{2}    + A{1}*p(4) + I*p(3);
        B=B*A{2}    + A{1}*p(2) + I*p(1);
        np=2;
	case 9
        B=A{3}*p(10) + A{2}*p(9) + A{1}*p(8) + I*p(7);
        B=B*A{3}    + A{2}*p(6) + A{1}*p(5) + I*p(4);
        B=B*A{3}    + A{2}*p(3) + A{1}*p(2) + I*p(1);
        np=2;
	case 12
        B=A{3}*p(13) + A{2}*p(12) + A{1}*p(11) + I*p(10) ;
        B=B*A{3}     + A{2}*p(9)  + A{1}*p(8)  + I*p(7);
        B=B*A{3}     + A{2}*p(6)  + A{1}*p(5)  + I*p(4);
        B=B*A{3}     + A{2}*p(3)  + A{1}*p(2)  + I*p(1);
        np=3;
	case 16
        B=A{4}*p(17) + A{3}*p(16) + A{2}*p(15) + A{1}*p(14) + I*p(13);
        B=A{4}*B     + A{3}*p(12) + A{2}*p(11) + A{1}*p(10) + I*p(9);
        B=A{4}*B     + A{3}*p(8)  + A{2}*p(7)  + A{1}*p(6)  + I*p(5);
        B=A{4}*B     + A{3}*p(4)  + A{2}*p(3)  + A{1}*p(2)  + I*p(1);
        np=3;
	case 20
        B=A{4}*p(21)  + A{3}*p(20) + A{2}*p(19) + A{1}*p(18) + I*p(17);
        B=A{4}*B      + A{3}*p(16) + A{2}*p(15) + A{1}*p(14) + I*p(13);
        B=A{4}*B      + A{3}*p(12) + A{2}*p(11) + A{1}*p(10) + I*p(9);
        B=A{4}*B      + A{3}*p(8)  + A{2}*p(7)  + A{1}*p(6)  + I*p(5);
        B=A{4}*B      + A{3}*p(4)  + A{2}*p(3)  + A{1}*p(2)  + I*p(1);    
        np=4;
end

end
    
