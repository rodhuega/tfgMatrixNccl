
function gen_matrices_toolbox_eigtoolv1(F,n,nd)
% gen_matrices_toolbox_eigtool(F,n,nd)
%
% Genera matrices de test de los paquetes Toolbox y Eigtool. Por cada 
% matriz se genera un fichero 
% .mat que almacena:
% - A:         Matriz del paquete Toolbox on Eigtool.
% - normA:     1-norma de la matriz A.
% - condFA:    Número de condición de la función matricial.
% - FA:        "Valor exacto" de F(A); es decir,función matricial aplicada
%               a la matriz A.
% - flag_error: indica si ha habido error o no en el cálculo de F(A).
%              -flag_error=0: No ha habido error
%              -flag_error=-1: No se ha podido calcular el "valor exacto"
%                              de F(A),pues dos cálculo sucesivos de Taylor  
%                              no handado el mismo valor (salvo errores 
%                              menores que el error unidad en doble precisión.
%              -Para valores distintos a los anteriores, indica que el
%               orden de la aproximación de Taylor que no se ha podido
%               calcular al variar el escalado
% Adicionalmente, se crea un fichero .m con el número total de ficheros
% mat generados. Cuando no ha sido posible calcular F(A), en el fichero .mat
% solo se almacena la variable flag_error; en otro caso, se almacenan  
% las variables indicadas anteriormente.
%
% Datos de entrada:
% - F:    Función a aplicar sobre la matriz (@exp, @cos, @cosh, ...). Por
%         el momento, sólo es válida para la exponencial y el coseno.
% - n:    Número de filas y columnas de la matriz. 
% - nd:   Número de dígitos vpa (32, 64, 128, 256).
%
% Ejemplo de invocación:
% gen_matrices_toolbox_eigtool(@exp,128,256)
%Version v1 25-6-2019
%Cambios respecto de la versión inicial:
%1.Cambio nobre variable e por flag_error
%2.Ahora se generan todas las matrices pero añadiendo la variable
%flag_error que indica si ha habido error, con su código de error, o no lo
%ha habido

test=1;
crea_mats(F,test,n,nd);
test=2;
crea_mats(F,test,n,nd)
end

function crea_mats(F,test,n,nd)
% crea_mats(F,test,n,nd)
%
% Genera matrices de test de los paquetes toolbox y eigtool. Por cada 
% matriz se genera un fichero 
% .mat que almacena:
% - A:      Matriz perteneciente al paquete Toolbox o Eigtool.
% - normA:  1-norma de la matriz A.
% - condFA: Número de condición de la matriz F(A).
% - FA:     Resultado de la función matricial aplicada sobre la matriz A.
% - flag_error: indica si ha habido error o no en el cálculo de F(A).
%              -flag_error=0: No ha habido error
%              -flag_error=-1: No se ha podido calcular el "valor exacto"
%                              pues dos cálculo sucesivos de Taylor no han 
%                              dado el mismo valor (salvo errores menores
%                              que el error unidad en doble precisión.
%              -Para valores distintos a los anteriores, indica que el
%               orden de la aproximación de Taylor que no se ha podido
%               calcular al variar el escalado

% Adicionalmente, se crea un fichero .m con el número de matrices
% generadas.
%
% Datos de entrada:
% - F:    Función a aplicar sobre la matriz (@exp, @cos, @cosh, ...). Por
%         el momento, sólo es válida para la exponencial y el coseno.
% - test: Decide si se generan matrices de las toolbox (1) o eigtool (2).
% - n:    Número de filas y columnas de la matriz. 
% - nd:   Número de dígitos vpa (32, 64, 128, 256).
%
% Ejemplo de invocación:
% crea_mats(@exp,1,128,256)

digits(nd);
im=9;%Inicia cálculo para m=30
if isequal(F,@exp)
	FT=@taylor_aux_exp;     
elseif isequal(F,@cos)
    FT=@taylor_aux_cos;
end
if test==1 % toolbox
    t0=sprintf('%s_toolbox_n%d_nd%d',func2str(F),n,nd);
    t=sprintf('%s/bad_matrix_toolbox.txt',t0);
    lm=52;
else       % eigtool
    t0=sprintf('%s_eigtool_n%d_nd%d',func2str(F),n,nd);
    t=sprintf('%s/bad_matrix_eigtool.txt',t0);
    f=eigtool_matrix();lm=length(f);
end

try
    warning off
    eval(sprintf('mkdir %s',t0));
catch
end
fileID = fopen(t,'w');
rng('default');
j=0;
for k=1:lm
    tic
    if test==1
        A=matrix(k,n);
    else
        A= feval(f{k},n);
    end
    [FA,normA,condFA,flag_error]=almacena_matrices(F,FT,im,A,nd);
    A=double(A);FA=double(FA);normA=double(normA);condFA=double(condFA);
    if flag_error==0
        j=j+1;
        t=sprintf('save %s/%s_%d.mat A FA normA condFA flag_error',t0,t0,k);
        eval(t);
    elseif flag_error==-1
        fprintf('Bad matrix %d: Dos aproximaciones consecutivas de Taylor siempre han dado error.\n',k);
        fprintf(fileID,'Bad matrix %d: Dos aproximaciones consecutivas de Taylor siempre han dado error.\n',k); 
        t=sprintf('save %s/%s_%d.mat flag_error',t0,t0,k);
        eval(t);
    else
        fprintf('Bad matrix %d: No se ha podido calcular Taylor, variando el escalado, para m=%d.\n',k,flag_error);
        fprintf(fileID,'Bad matrix %d: No se ha podido calcular Taylor, variando el escalado, para m=%d.\n',k,flag_error); 
        t=sprintf('save %s/%s_%d.mat flag_error',t0,t0,k);
        eval(t);
    end
    fprintf('Test %s: De %d matrices se han podido calcular hasta ahora %d matrices.\n',t0,k,j)
end
fclose(fileID);
t=sprintf('%s/%s.m',t0,t0);
fileID = fopen(t,'w');
fprintf(fileID,'nmat=%d;',lm);
fclose(fileID);
end

function [FA,normA,condFA,flag_error]=almacena_matrices(F,FT,im,A,nd)
M= [2 4 6 9 12 16 20 25 30 36 42 49 56 64];
q=[2 2 3 3  4  4  5  5  6  6  7  7  8  8];

digits(nd);
A=vpa(A);
normA=[];
condFA=[];
FA=[];
if isequal(F,@exp)
    pA{1}=A;
elseif isequal(F,@cos)
    pA{1}=vpa(A*A);  
end   
for i=2:q(im)
    pA{i}=vpa(pA{1}*pA{i-1});
    if isinf(norm(pA{i}))
        flag_error=M(im);
        return
    end
end
[FA0,flag_error,~,~]=FT(pA,im,nd);
if flag_error
	return
end
fin=0;
while fin==0&&im<14 %im=14 corresponde a m=64 
    im=im+1;
    if mod(im,2)==1
       pA{q(im)}=vpa(pA{1}*pA{q(im-1)}); 
    end
    [FA,flag_error,~,~]=FT(pA,im,nd);
    if flag_error
        return
    end
    er=norm(FA-FA0,1)/norm(FA);
    if er<sym(eps/2)
        fin=1;
    end
    FA0=FA;
end
if fin==1
    normA=norm(A,1);
    condFA=funm_condest1(A,F);
else
    flag_error=-1;
end
end

function [B,flag_error,s,er]=taylor_aux_exp(pA,im,nd)
M= [2 4 6 9 12 16 20 25 30 36 42 49 56 64];
q=[2 2 3 3  4  4  5  5  6  6  7  7  8  8];
digits(nd);
n=size(pA{1},1);
er=0;
p =[1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000, 2432902008176640000, 51090942171709440000, 1124000727777607680000, 25852016738884978212864, 620448401733239409999872, 15511210043330986055303168, 403291461126605650322784256, 10888869450418351940239884288, 304888344611713836734530715648, 8841761993739700772720181510144, 265252859812191032188804700045312, 8222838654177922430198509928972288, 263130836933693517766352317727113216, 8683317618811885938715673895318323200, 295232799039604119555149671006000381952, 10333147966386144222209170348167175077888, 371993326789901177492420297158468206329856, 13763753091226343102992036262845720547033088, 523022617466601037913697377988137380787257344, 20397882081197441587828472941238084160318341120, 815915283247897683795548521301193790359984930816, 33452526613163802763987613764361857922667238129664, 1405006117752879788779635797590784832178972610527232, 60415263063373834074440829285578945930237590418489344, 2658271574788448529134213028096241889243150262529425408, 119622220865480188574992723157469373503186265858579103744, 5502622159812088456668950435842974564586819473162983440384, 258623241511168177673491006652997026552325199826237836492800, 12413915592536072528327568319343857274511609591659416151654400, 608281864034267522488601608116731623168777542102418391010639872, 30414093201713375576366966406747986832057064836514787179557289984, 1551118753287382189470754582685817365323346291853046617899802820608, 80658175170943876845634591553351679477960544579306048386139594686464, 4274883284060025484791254765342395718256495012315011061486797910441984, 230843697339241379243718839060267085502544784965628964557765331531071488, 12696403353658276446882823840816011312245221598828319560272916152712167424, 710998587804863481025438135085696633485732409534385895375283304551881375744, 40526919504877220527556156789809444757511993541235911846782577699372834750464, 2350561331282878906297796280456247634956966273955390268712005058924708557225984, 138683118545689864933221185143853352853533809868133429504739525869642019130834944, 8320987112741391580056396102959641077457945541076708813599085350531187384917164032, 507580213877224835833540161373088490724281389843871724559898414118829028410677788672, 31469973260387939390320343330721249710233204778005956144519390914718240063804258910208, 1982608315404440084965732774767545707658109829136018902789196017241837351744329178152960, 126886932185884165437806897585122925290119029064705209778508545103477590511637067401789440];
flag_error=M(im);
B=ps_d(p,M(im),pA,n);
s=0;
while flag_error==M(im)&&s<=15
    B0=B;
    s=s+1;
    for i=1:q(im)
        pB{i}=vpa(pA{i}/2^(i*s));
    end
    B=ps_d(p,M(im),pB,n);
    if max(max(isnan(B)))||max(max(isinf(B)))
        return
    end
    for k=1:s
        B=vpa(B*B);
    end
    if max(max(isnan(B)))||max(max(isinf(B)))
        return
    end
    er=norm(double(B-B0))/norm(double(B));
    %fprintf('s=%d: er=%g\n',s,er);
    if er<sym(eps/2)
        flag_error=0;
    end
end
end

function [B,flag_error,s,er]=taylor_aux_cos(pA,im,nd)
M= [2 4 6 9 12 16 20 25 30 36 42 49 56 64];
q=[2 2 3 3  4  4  5  5  6  6  7  7  8  8];
digits(nd);
n=size(pA{1},1);
er=0;
p=[-2, 24, -720, 40320, -3628800, 479001600, -87178291200, 20922789888000, -6402373705728000, 2432902008176640000, -1124000727777607680000, 620448401733239439360000, -403291461126605635584000000, 304888344611713860501504000000, -265252859812191058636308480000000, 263130836933693530167218012160000000, -295232799039604140847618609643520000000, 371993326789901217467999448150835200000000, -523022617466601111760007224100074291200000000, 815915283247897734345611269596115894272000000000, -1405006117752879898543142606244511569936384000000000, 2658271574788448768043625811014615890319638528000000000, -5502622159812088949850305428800254892961651752960000000000, 12413915592536072670862289047373375038521486354677760000000000, -30414093201713378043612608166064768844377641568960512000000000000, 80658175170943878571660636856403766975289505440883277824000000000000, -230843697339241380472092742683027581083278564571807941132288000000000000, 710998587804863451854045647463724949736497978881168458687447040000000000000, -2350561331282878571829474910515074683828862318181142924420699914240000000000000, 8320987112741390144276341183223364380754172606361245952449277696409600000000000000, -31469973260387937525653122354950764088012280797258232192163168247821107200000000000000, 126886932185884164103433389335161480802865516174545192198801894375214704230400000000000000, -544344939077443064003729240247842752644293064388798874532860126869671081148416000000000000000, 2480035542436830599600990418569171581047399201355367672371710738018221445712183296000000000000000, -11978571669969891796072783721689098736458938142546425857555362864628009582789845319680000000000000000, 61234458376886086861524070385274672740778091784697328983823014963978384987221689274204160000000000000000, -330788544151938641225953028221253782145683251820934971170611926835411235700971565459250872320000000000000000, 1885494701666050254987932260861146558230394535379329335672487982961844043495537923117729972224000000000000000000, -11324281178206297831457521158732046228731749579488251990048962825668835325234200766245086213177344000000000000000000, 71569457046263802294811533723186532165584657342365752577109445058227039255480148842668944867280814080000000000000000000, -475364333701284174842138206989404946643813294067993328617160934076743994734899148613007131808479167119360000000000000000000, 3314240134565353266999387579130131288000666286242049487118846032383059131291716864129885722968716753156177920000000000000000000];
flag_error=im;
B=ps_d(p,M(im),pA,n);
s=0;
while flag_error==M(im)&&s<=15
    B0=B;
    s=s+1;
    for i=1:q(im)
        pB{i}=vpa(pA{i}/4^(i*s));
    end
    B=ps_d(p,M(im),pB,n);
    for k=1:s
        B=vpa(2*B*B-eye(n));
    end
	if max(max(isnan(B)))||max(max(isinf(B)))
        return
    end
    er=norm(B-B0,1)/norm(B,1);
    %fprintf('s=%d: er=%g\n',s,er);
    if er<sym(eps/2)
        flag_error=0;
    end
end
end

function B=ps_d(p,m,pA,n)
I=vpa(eye(n));
switch m
    case 2 %q=2, c=1
        B=vpa(pA{2}/p(2) +pA{1} + I);
	case 4 %q=2, c=2
        B=vpa(pA{2}/p(4)  + pA{1}/p(3) + I/p(2));
        B=vpa(B*pA{2}     + pA{1}   + I);
	case 6 %q=3, c=2
        B=vpa(pA{3}/p(6) + pA{2}/p(5) + pA{1}/p(4) + I/p(3));
        B=vpa(B*pA{3}    + pA{2}/p(2) + pA{1}      + I);
	case 9 %q=3, c=3
        B=vpa(pA{3}/p(9) + pA{2}/p(8) + pA{1}/p(7) + I/p(6));
        B=vpa(B*pA{3}    + pA{2}/p(5) + pA{1}/p(4) + I/p(3));
        B=vpa(B*pA{3}    + pA{2}/p(2) + pA{1}      + I);
	case 12 %q=4, c=3
        B=vpa(pA{4}/p(12) + pA{3}/p(11) + pA{2}/p(10) + pA{1}/p(9) + I/p(8));
        B=vpa(B*pA{4}     + pA{3}/p(7)  + pA{2}/p(6)  + pA{1}/p(5) + I/p(4));
        B=vpa(B*pA{4}     + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}      + I);
	case 16 %q=4, c=4
        B=vpa(pA{4}/p(16) + pA{3}/p(15) + pA{2}/p(14) + pA{1}/p(13) + I/p(12));
        B=vpa(pA{4}*B     + pA{3}/p(11) + pA{2}/p(10) + pA{1}/p(9)  + I/p(8));
        B=vpa(pA{4}*B     + pA{3}/p(7)  + pA{2}/p(6)  + pA{1}/p(5)  + I/p(4));
        B=vpa(pA{4}*B     + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}       + I);
	case 20 %q=5, c=4
        B=vpa(pA{5}/p(20) + pA{4}/p(19) + pA{3}/p(18) + pA{2}/p(17) + pA{1}/p(16) + I/p(15));
        B=vpa(pA{5}*B     + pA{4}/p(14) + pA{3}/p(13) + pA{2}/p(12) + pA{1}/p(11) + I/p(10));
        B=vpa(pA{5}*B     + pA{4}/p(9)  + pA{3}/p(8)  + pA{2}/p(7)  + pA{1}/p(6)  + I/p(5));
        B=vpa(pA{5}*B     + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}       + I);
	case 25 %q=5, c=5
        B=vpa(pA{5}/p(25) + pA{4}/p(24) + pA{3}/p(23) + pA{2}/p(22) + pA{1}/p(21) + I/p(20));
        B=vpa(pA{5}*B     + pA{4}/p(19) + pA{3}/p(18) + pA{2}/p(17) + pA{1}/p(16) + I/p(15));
        B=vpa(pA{5}*B     + pA{4}/p(14) + pA{3}/p(13) + pA{2}/p(12) + pA{1}/p(11) + I/p(10));
        B=vpa(pA{5}*B     + pA{4}/p(9)  + pA{3}/p(8)  + pA{2}/p(7)  + pA{1}/p(6)  + I/p(5));
        B=vpa(pA{5}*B     + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}       + I);
	case 30 %q=6, c=5
        B=vpa(pA{6}/p(30) + pA{5}/p(29) + pA{4}/p(28) + pA{3}/p(27) + pA{2}/p(26) + pA{1}/p(25) + I/p(24));
        B=vpa(pA{6}*B     + pA{5}/p(23) + pA{4}/p(22) + pA{3}/p(21) + pA{2}/p(20) + pA{1}/p(19) + I/p(18));
        B=vpa(pA{6}*B     + pA{5}/p(17) + pA{4}/p(16) + pA{3}/p(15) + pA{2}/p(14) + pA{1}/p(13) + I/p(12));
        B=vpa(pA{6}*B     + pA{5}/p(11) + pA{4}/p(10) + pA{3}/p(9)  + pA{2}/p(8)  + pA{1}/p(7)  + I/p(6));
        B=vpa(pA{6}*B     + pA{5}/p(5)  + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I);
	case 36 %q=6
        B=vpa(pA{6}/p(36) + pA{5}/p(35) + pA{4}/p(34) + pA{3}/p(33) + pA{2}/p(32) + pA{1}/p(31) + I/p(30));
        B=vpa(pA{6}*B     + pA{5}/p(29) + pA{4}/p(28) + pA{3}/p(27) + pA{2}/p(26) + pA{1}/p(25) + I/p(24));
        B=vpa(pA{6}*B     + pA{5}/p(23) + pA{4}/p(22) + pA{3}/p(21) + pA{2}/p(20) + pA{1}/p(19) + I/p(18));
        B=vpa(pA{6}*B     + pA{5}/p(17) + pA{4}/p(16) + pA{3}/p(15) + pA{2}/p(14) + pA{1}/p(13) + I/p(12));
        B=vpa(pA{6}*B     + pA{5}/p(11) + pA{4}/p(10) + pA{3}/p(9)  + pA{2}/p(8)  + pA{1}/p(7)  + I/p(6));
        B=vpa(pA{6}*B     + pA{5}/p(5)  + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I);
	case 42 %q=7
        B=vpa(pA{7}/p(42) + pA{6}/p(41) + pA{5}/p(40) + pA{4}/p(39) + pA{3}/p(38) + pA{2}/p(37) + pA{1}/p(36) + I/p(35));
        B=vpa(pA{7}*B     + pA{6}/p(34) + pA{5}/p(33) + pA{4}/p(32) + pA{3}/p(31) + pA{2}/p(30) + pA{1}/p(29) + I/p(28));
        B=vpa(pA{7}*B     + pA{6}/p(27) + pA{5}/p(26) + pA{4}/p(25) + pA{3}/p(24) + pA{2}/p(23) + pA{1}/p(22) + I/p(21));
        B=vpa(pA{7}*B     + pA{6}/p(20) + pA{5}/p(19) + pA{4}/p(18) + pA{3}/p(17) + pA{2}/p(16) + pA{1}/p(15) + I/p(14));
        B=vpa(pA{7}*B     + pA{6}/p(13) + pA{5}/p(12) + pA{4}/p(11) + pA{3}/p(10) + pA{2}/p(9)  + pA{1}/p(8)  + I/p(7));
        B=vpa(pA{7}*B     + pA{6}/p(6)  + pA{5}/p(5)  + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I);
	case 49 %q=7
        B=vpa(pA{7}/p(49) + pA{6}/p(48) + pA{5}/p(47) + pA{4}/p(46) + pA{3}/p(45) + pA{2}/p(44) + pA{1}/p(43) + I/p(42));
        B=vpa(pA{7}*B     + pA{6}/p(41) + pA{5}/p(40) + pA{4}/p(39) + pA{3}/p(38) + pA{2}/p(37) + pA{1}/p(36) + I/p(35));
        B=vpa(pA{7}*B     + pA{6}/p(34) + pA{5}/p(33) + pA{4}/p(32) + pA{3}/p(31) + pA{2}/p(30) + pA{1}/p(29) + I/p(28));
        B=vpa(pA{7}*B     + pA{6}/p(27) + pA{5}/p(26) + pA{4}/p(25) + pA{3}/p(24) + pA{2}/p(23) + pA{1}/p(22) + I/p(21));
        B=vpa(pA{7}*B     + pA{6}/p(20) + pA{5}/p(19) + pA{4}/p(18) + pA{3}/p(17) + pA{2}/p(16) + pA{1}/p(15) + I/p(14));
        B=vpa(pA{7}*B     + pA{6}/p(13) + pA{5}/p(12) + pA{4}/p(11) + pA{3}/p(10) + pA{2}/p(9)  + pA{1}/p(8)  + I/p(7));
        B=vpa(pA{7}*B     + pA{6}/p(6)  + pA{5}/p(5)  + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I); 
    case 56 %q=8
        B=vpa(pA{8}/p(56) + pA{7}/p(55) + pA{6}/p(54) + pA{5}/p(53) + pA{4}/p(52) + pA{3}/p(51) + pA{2}/p(50) + pA{1}/p(49) + I/p(48));
        B=vpa(pA{8}*B  + pA{7}/p(47) + pA{6}/p(46) + pA{5}/p(45) + pA{4}/p(44) + pA{3}/p(43) + pA{2}/p(42) + pA{1}/p(41) + I/p(40));
        B=vpa(pA{8}*B  + pA{7}/p(39) + pA{6}/p(38) + pA{5}/p(37) + pA{4}/p(36) + pA{3}/p(35) + pA{2}/p(34) + pA{1}/p(33) + I/p(32));
        B=vpa(pA{8}*B  + pA{7}/p(31) + pA{6}/p(30) + pA{5}/p(29) + pA{4}/p(28) + pA{3}/p(27) + pA{2}/p(26) + pA{1}/p(25) + I/p(24));
        B=vpa(pA{8}*B  + pA{7}/p(23) + pA{6}/p(22) + pA{5}/p(21) + pA{4}/p(20) + pA{3}/p(19) + pA{2}/p(18) + pA{1}/p(17) + I/p(16));
        B=vpa(pA{8}*B  + pA{7}/p(15) + pA{6}/p(14) + pA{5}/p(13) + pA{4}/p(12) + pA{3}/p(11) + pA{2}/p(10) + pA{1}/p(9)  + I/p(8));
        B=vpa(pA{8}*B  + pA{7}/p(7)  + pA{6}/p(6)  + pA{5}/p(5)  + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I); 
    case 64 %q=8
        B=vpa(pA{8}/p(64) + pA{7}/p(63) + pA{6}/p(62) + pA{5}/p(61) + pA{4}/p(60) + pA{3}/p(59) + pA{2}/p(58) + pA{1}/p(57) + I/p(56));
        B=vpa(pA{8}*B  + pA{7}/p(55) + pA{6}/p(54) + pA{5}/p(53) + pA{4}/p(52) + pA{3}/p(51) + pA{2}/p(50) + pA{1}/p(49) + I/p(48));
        B=vpa(pA{8}*B  + pA{7}/p(47) + pA{6}/p(46) + pA{5}/p(45) + pA{4}/p(44) + pA{3}/p(43) + pA{2}/p(42) + pA{1}/p(41) + I/p(40));
        B=vpa(pA{8}*B  + pA{7}/p(39) + pA{6}/p(38) + pA{5}/p(37) + pA{4}/p(36) + pA{3}/p(35) + pA{2}/p(34) + pA{1}/p(33) + I/p(32));
        B=vpa(pA{8}*B  + pA{7}/p(31) + pA{6}/p(30) + pA{5}/p(29) + pA{4}/p(28) + pA{3}/p(27) + pA{2}/p(26) + pA{1}/p(25) + I/p(24));
        B=vpa(pA{8}*B  + pA{7}/p(23) + pA{6}/p(22) + pA{5}/p(21) + pA{4}/p(20) + pA{3}/p(19) + pA{2}/p(18) + pA{1}/p(17) + I/p(16));
        B=vpa(pA{8}*B  + pA{7}/p(15) + pA{6}/p(14) + pA{5}/p(13) + pA{4}/p(12) + pA{3}/p(11) + pA{2}/p(10) + pA{1}/p(9)  + I/p(8));
        B=vpa(pA{8}*B  + pA{7}/p(7)  + pA{6}/p(6)  + pA{5}/p(5)  + pA{4}/p(4)  + pA{3}/p(3)  + pA{2}/p(2)  + pA{1}/p(1)  + I); 
end

end

