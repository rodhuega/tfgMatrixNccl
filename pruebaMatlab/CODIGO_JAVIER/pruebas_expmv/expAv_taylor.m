function [w,m,sm,npm]=expAv_taylor(A,v,imax)
n=size(A,1);

% Theta backward relativo para la funci�n exponencial (nd=128, nt=100): 
zeta=[1.4382525968043369  % m=20
      2.4285825244428265  % m=25
      3.5396663487436895  % m=30
      4.9729156261919814  % m=36
      6.4756827360799845  % m=42
      8.2848536298039175  % m=49
      10.133428317897478  % m=56
      12.277837026932959];% m=64
  
M=[20 25 30 6 42 49 56 64];
p1=[5 5 6 6 7 7 8 8];
p2=[4 5 5 6 6 7 7 8];
pA{1}=A;pA{2}=A*A;pA{3}=pA{2}*A;pA{4}=pA{3}*A;
nofin=1;i=0;
while nofin&&i<imax
    i=i+1;
    j=ceil(sqrt(M(i)));
    if sqrt(M(i))>floor(sqrt(M(i)))
        pA{j}=pA{j-1}*A;
    end
    alfa(i)=max(norm1pp(pA{p1(i)},p1(i),A)^(1/(M(i)+1)),norm1pp(pA{p1(i)},p2(i),pA{2})^(1/(M(i)+2)));
    if alfa(i)<zeta(i)
        nofin=0;     
    end
end
if nofin==0
    sm=0;
    m=M(i);
else
    for j=1:i
        s(j)=ceil(max(0,log2(alfa(j)/zeta(j))));
    end
    if s(i)>s(i-1)
        m=M(i-1);
        sm=s(i-1);
    else
        m=M(i);
        sm=s(i);
    end
end  
[A,npm]=ps_d(pA,m,sm);
npm=n*npm+1;
w=A*v;
end

function [B,np]=ps_d(pA,m,s)
n = size(pA{1},1); 
I = eye(n);
p =[1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000, 2432902008176640000, 51090942171709440000, 1124000727777607680000, 25852016738884978212864, 620448401733239409999872, 15511210043330986055303168, 403291461126605650322784256, 10888869450418351940239884288, 304888344611713836734530715648, 8841761993739700772720181510144, 265252859812191032188804700045312, 8222838654177922430198509928972288, 263130836933693517766352317727113216, 8683317618811885938715673895318323200, 295232799039604119555149671006000381952, 10333147966386144222209170348167175077888, 371993326789901177492420297158468206329856, 13763753091226343102992036262845720547033088, 523022617466601037913697377988137380787257344, 20397882081197441587828472941238084160318341120, 815915283247897683795548521301193790359984930816, 33452526613163802763987613764361857922667238129664, 1405006117752879788779635797590784832178972610527232, 60415263063373834074440829285578945930237590418489344, 2658271574788448529134213028096241889243150262529425408, 119622220865480188574992723157469373503186265858579103744, 5502622159812088456668950435842974564586819473162983440384, 258623241511168177673491006652997026552325199826237836492800, 12413915592536072528327568319343857274511609591659416151654400, 608281864034267522488601608116731623168777542102418391010639872, 30414093201713375576366966406747986832057064836514787179557289984, 1551118753287382189470754582685817365323346291853046617899802820608, 80658175170943876845634591553351679477960544579306048386139594686464, 4274883284060025484791254765342395718256495012315011061486797910441984, 230843697339241379243718839060267085502544784965628964557765331531071488, 12696403353658276446882823840816011312245221598828319560272916152712167424, 710998587804863481025438135085696633485732409534385895375283304551881375744, 40526919504877220527556156789809444757511993541235911846782577699372834750464, 2350561331282878906297796280456247634956966273955390268712005058924708557225984, 138683118545689864933221185143853352853533809868133429504739525869642019130834944, 8320987112741391580056396102959641077457945541076708813599085350531187384917164032, 507580213877224835833540161373088490724281389843871724559898414118829028410677788672, 31469973260387939390320343330721249710233204778005956144519390914718240063804258910208, 1982608315404440084965732774767545707658109829136018902789196017241837351744329178152960, 126886932185884165437806897585122925290119029064705209778508545103477590511637067401789440];

for i=1:length(pA)
    pA{i}=pA{i}/2^(i*s);
end
switch m
    case 2 %q=2, c=1
        np=1;
        B=pA{2}/p(2) +pA{1} + I;
	case 4 %q=2, c=2
        np=2;
        B=pA{2}/p(4)  + pA{1}/p(3) + I/p(2);
        B=B*pA{2}     + pA{1}   + I;
	case 6 %q=3, c=2
        np=3;
        B=pA{3}/p(6) + pA{2}/p(5) + pA{1}/p(4) + I/p(3);
        B=B*pA{3}    + pA{2}/p(2) + pA{1}      + I;
	case 9 %q=3, c=3
        np=4;
        B=pA{3}/p(9) + pA{2}/p(8) + pA{1}/p(7) + I/p(6);
        B=B*pA{3}    + pA{2}/p(5) + pA{1}/p(4) + I/p(3);
        B=B*pA{3}    + pA{2}/p(2) + pA{1}      + I;
	case 12 %q=4, c=3
        np=5;
        B=pA{4}/p(12) + pA{3}/p(11) + pA{2}/p(10) + pA{1}/p(9) + I/p(8);
        B=B*pA{4}     + pA{3}/p(7)  + pA{2}/p(6)  + pA{1}/p(5) + I/p(4);
        B=B*pA{4}     + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}      + I;
	case 16 %q=4, c=4
        np=6;
        B=pA{4}/p(16) + pA{3}/p(15) + pA{2}/p(14) + pA{1}/p(13) + I/p(12);
        B=pA{4}*B     + pA{3}/p(11) + pA{2}/p(10) + pA{1}/p(9)  + I/p(8);
        B=pA{4}*B     + pA{3}/p(7)  + pA{2}/p(6)  + pA{1}/p(5)  + I/p(4);
        B=pA{4}*B     + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}       + I;
	case 20 %q=5, c=4
        np=7;
        B=pA{5}/p(20) + pA{4}/p(19) + pA{3}/p(18) + pA{2}/p(17) + pA{1}/p(16) + I/p(15);
        B=pA{5}*B     + pA{4}/p(14) + pA{3}/p(13) + pA{2}/p(12) + pA{1}/p(11) + I/p(10);
        B=pA{5}*B     + pA{4}/p(9)  + pA{3}/p(8)  + pA{2}/p(7)  + pA{1}/p(6)  + I/p(5);
        B=pA{5}*B     + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}       + I;
	case 25 %q=5, c=5
        np=8;
        B=pA{5}/p(25) + pA{4}/p(24) + pA{3}/p(23) + pA{2}/p(22) + pA{1}/p(21) + I/p(20);
        B=pA{5}*B     + pA{4}/p(19) + pA{3}/p(18) + pA{2}/p(17) + pA{1}/p(16) + I/p(15);
        B=pA{5}*B     + pA{4}/p(14) + pA{3}/p(13) + pA{2}/p(12) + pA{1}/p(11) + I/p(10);
        B=pA{5}*B     + pA{4}/p(9)  + pA{3}/p(8)  + pA{2}/p(7)  + pA{1}/p(6)  + I/p(5);
        B=pA{5}*B     + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}       + I;
	case 30 %q=6, c=5
        np=9;
        B=pA{6}/p(30) + pA{5}/p(29) + pA{4}/p(28) + pA{3}/p(27) + pA{2}/p(26) + pA{1}/p(25) + I/p(24);
        B=pA{6}*B     + pA{5}/p(23) + pA{4}/p(22) + pA{3}/p(21) + pA{2}/p(20) + pA{1}/p(19) + I/p(18);
        B=pA{6}*B     + pA{5}/p(17) + pA{4}/p(16) + pA{3}/p(15) + pA{2}/p(14) + pA{1}/p(13) + I/p(12);
        B=pA{6}*B     + pA{5}/p(11) + pA{4}/p(10) + pA{3}/p(9)  + pA{2}/p(8)  + pA{1}/p(7)  + I/p(6);
        B=pA{6}*B     + pA{5}/p(5)  + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I;
	case 36 %q=6
        np=10;
        B=pA{6}/p(36) + pA{5}/p(35) + pA{4}/p(34) + pA{3}/p(33) + pA{2}/p(32) + pA{1}/p(31) + I/p(30);
        B=pA{6}*B     + pA{5}/p(29) + pA{4}/p(28) + pA{3}/p(27) + pA{2}/p(26) + pA{1}/p(25) + I/p(24);
        B=pA{6}*B     + pA{5}/p(23) + pA{4}/p(22) + pA{3}/p(21) + pA{2}/p(20) + pA{1}/p(19) + I/p(18);
        B=pA{6}*B     + pA{5}/p(17) + pA{4}/p(16) + pA{3}/p(15) + pA{2}/p(14) + pA{1}/p(13) + I/p(12);
        B=pA{6}*B     + pA{5}/p(11) + pA{4}/p(10) + pA{3}/p(9)  + pA{2}/p(8)  + pA{1}/p(7)  + I/p(6);
        B=pA{6}*B     + pA{5}/p(5)  + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I;
	case 42 %q=7
        np=11;
        B=pA{7}/p(42) + pA{6}/p(41) + pA{5}/p(40) + pA{4}/p(39) + pA{3}/p(38) + pA{2}/p(37) + pA{1}/p(36) + I/p(35);
        B=pA{7}*B     + pA{6}/p(34) + pA{5}/p(33) + pA{4}/p(32) + pA{3}/p(31) + pA{2}/p(30) + pA{1}/p(29) + I/p(28);
        B=pA{7}*B     + pA{6}/p(27) + pA{5}/p(26) + pA{4}/p(25) + pA{3}/p(24) + pA{2}/p(23) + pA{1}/p(22) + I/p(21);
        B=pA{7}*B     + pA{6}/p(20) + pA{5}/p(19) + pA{4}/p(18) + pA{3}/p(17) + pA{2}/p(16) + pA{1}/p(15) + I/p(14);
        B=pA{7}*B     + pA{6}/p(13) + pA{5}/p(12) + pA{4}/p(11) + pA{3}/p(10) + pA{2}/p(9)  + pA{1}/p(8)  + I/p(7);
        B=pA{7}*B     + pA{6}/p(6)  + pA{5}/p(5)  + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I;
	case 49 %q=7
        np=12;
        B=pA{7}/p(49) + pA{6}/p(48) + pA{5}/p(47) + pA{4}/p(46) + pA{3}/p(45) + pA{2}/p(44) + pA{1}/p(43) + I/p(42);
        B=pA{7}*B     + pA{6}/p(41) + pA{5}/p(40) + pA{4}/p(39) + pA{3}/p(38) + pA{2}/p(37) + pA{1}/p(36) + I/p(35);
        B=pA{7}*B     + pA{6}/p(34) + pA{5}/p(33) + pA{4}/p(32) + pA{3}/p(31) + pA{2}/p(30) + pA{1}/p(29) + I/p(28);
        B=pA{7}*B     + pA{6}/p(27) + pA{5}/p(26) + pA{4}/p(25) + pA{3}/p(24) + pA{2}/p(23) + pA{1}/p(22) + I/p(21);
        B=pA{7}*B     + pA{6}/p(20) + pA{5}/p(19) + pA{4}/p(18) + pA{3}/p(17) + pA{2}/p(16) + pA{1}/p(15) + I/p(14);
        B=pA{7}*B     + pA{6}/p(13) + pA{5}/p(12) + pA{4}/p(11) + pA{3}/p(10) + pA{2}/p(9)  + pA{1}/p(8)  + I/p(7);
        B=pA{7}*B     + pA{6}/p(6)  + pA{5}/p(5)  + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I; 
    case 56 %q=8
        np=13;
        B=pA{8}/p(56) + pA{7}/p(55) + pA{6}/p(54) + pA{5}/p(53) + pA{4}/p(52) + pA{3}/p(51) + pA{2}/p(50) + pA{1}/p(49) + I/p(48);
        B=   pA{8}*B  + pA{7}/p(47) + pA{6}/p(46) + pA{5}/p(45) + pA{4}/p(44) + pA{3}/p(43) + pA{2}/p(42) + pA{1}/p(41) + I/p(40);
        B=   pA{8}*B  + pA{7}/p(39) + pA{6}/p(38) + pA{5}/p(37) + pA{4}/p(36) + pA{3}/p(35) + pA{2}/p(34) + pA{1}/p(33) + I/p(32);
        B=   pA{8}*B  + pA{7}/p(31) + pA{6}/p(30) + pA{5}/p(29) + pA{4}/p(28) + pA{3}/p(27) + pA{2}/p(26) + pA{1}/p(25) + I/p(24);
        B=   pA{8}*B  + pA{7}/p(23) + pA{6}/p(22) + pA{5}/p(21) + pA{4}/p(20) + pA{3}/p(19) + pA{2}/p(18) + pA{1}/p(17) + I/p(16);
        B=   pA{8}*B  + pA{7}/p(15) + pA{6}/p(14) + pA{5}/p(13) + pA{4}/p(12) + pA{3}/p(11) + pA{2}/p(10) + pA{1}/p(9)  + I/p(8);
        B=   pA{8}*B  + pA{7}/p(7)  + pA{6}/p(6)  + pA{5}/p(5)  + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I; 
    case 64 %q=8
        np=14;
        B=pA{8}/p(64) + pA{7}/p(63) + pA{6}/p(62) + pA{5}/p(61) + pA{4}/p(60) + pA{3}/p(59) + pA{2}/p(58) + pA{1}/p(57) + I/p(56);
        B=   pA{8}*B  + pA{7}/p(55) + pA{6}/p(54) + pA{5}/p(53) + pA{4}/p(52) + pA{3}/p(51) + pA{2}/p(50) + pA{1}/p(49) + I/p(48);
        B=   pA{8}*B  + pA{7}/p(47) + pA{6}/p(46) + pA{5}/p(45) + pA{4}/p(44) + pA{3}/p(43) + pA{2}/p(42) + pA{1}/p(41) + I/p(40);
        B=   pA{8}*B  + pA{7}/p(39) + pA{6}/p(38) + pA{5}/p(37) + pA{4}/p(36) + pA{3}/p(35) + pA{2}/p(34) + pA{1}/p(33) + I/p(32);
        B=   pA{8}*B  + pA{7}/p(31) + pA{6}/p(30) + pA{5}/p(29) + pA{4}/p(28) + pA{3}/p(27) + pA{2}/p(26) + pA{1}/p(25) + I/p(24);
        B=   pA{8}*B  + pA{7}/p(23) + pA{6}/p(22) + pA{5}/p(21) + pA{4}/p(20) + pA{3}/p(19) + pA{2}/p(18) + pA{1}/p(17) + I/p(16);
        B=   pA{8}*B  + pA{7}/p(15) + pA{6}/p(14) + pA{5}/p(13) + pA{4}/p(12) + pA{3}/p(11) + pA{2}/p(10) + pA{1}/p(9)  + I/p(8);
        B=   pA{8}*B  + pA{7}/p(7)  + pA{6}/p(6)  + pA{5}/p(5)  + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I; 
end
for i=1:s
    B=B*B;
end
np=np+s;
end

function [A,np]= ps_d_(pA,m,s)
p =[ 1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000, 2432902008176640000, 51090942171709440000, 1124000727777607680000, 25852016738884978212864, 620448401733239409999872, 15511210043330986055303168, 403291461126605650322784256, 10888869450418351940239884288, 304888344611713836734530715648, 8841761993739700772720181510144, 265252859812191032188804700045312, 8222838654177922430198509928972288, 263130836933693517766352317727113216, 8683317618811885938715673895318323200, 295232799039604119555149671006000381952, 10333147966386144222209170348167175077888, 371993326789901177492420297158468206329856, 13763753091226343102992036262845720547033088, 523022617466601037913697377988137380787257344, 20397882081197441587828472941238084160318341120, 815915283247897683795548521301193790359984930816, 33452526613163802763987613764361857922667238129664, 1405006117752879788779635797590784832178972610527232, 60415263063373834074440829285578945930237590418489344, 2658271574788448529134213028096241889243150262529425408, 119622220865480188574992723157469373503186265858579103744, 5502622159812088456668950435842974564586819473162983440384, 258623241511168177673491006652997026552325199826237836492800, 12413915592536072528327568319343857274511609591659416151654400, 608281864034267522488601608116731623168777542102418391010639872, 30414093201713375576366966406747986832057064836514787179557289984, 1551118753287382189470754582685817365323346291853046617899802820608, 80658175170943876845634591553351679477960544579306048386139594686464, 4274883284060025484791254765342395718256495012315011061486797910441984, 230843697339241379243718839060267085502544784965628964557765331531071488, 12696403353658276446882823840816011312245221598828319560272916152712167424, 710998587804863481025438135085696633485732409534385895375283304551881375744, 40526919504877220527556156789809444757511993541235911846782577699372834750464, 2350561331282878906297796280456247634956966273955390268712005058924708557225984, 138683118545689864933221185143853352853533809868133429504739525869642019130834944, 8320987112741391580056396102959641077457945541076708813599085350531187384917164032, 507580213877224835833540161373088490724281389843871724559898414118829028410677788672, 31469973260387939390320343330721249710233204778005956144519390914718240063804258910208, 1982608315404440084965732774767545707658109829136018902789196017241837351744329178152960, 126886932185884165437806897585122925290119029064705209778508545103477590511637067401789440];
q=length(pA);
r=m-(q-1)^2;
m=m+1;
n=size(pA,1);
I=eye(n);
%r=(m-1)/q;
for k=1:q
    pA{k}=pA{k}/2^(s*k);
end
    
A=I/p(m-q);
for i=1:q
   A=A+pA{i}/p(m-q+i); 
end
for k=1:r-1
    A=A*pA{q}+I/p(m-q*(k+1));
    for i=1:q-1
        A=A+pA{i}/p(m-q*(k+1)+i); 
    end  
end
for k=1:s
    A=A*A;
end
np=q+r-2+s;
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