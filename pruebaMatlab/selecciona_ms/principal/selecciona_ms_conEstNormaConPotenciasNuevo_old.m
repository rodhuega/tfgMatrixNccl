function [m,sm,pA,nProd]=selecciona_ms_conEstNormaConPotenciasNuevo(f,metodo_f,plataforma,A,mmin,mmax)
% [m,sm,pA,nProd]=selecciona_ms_conEstNormaConPotenciasNuevo(f,metodo_f,plataforma,A,mmin,mmax)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz estimando la norma de las potencias de la matriz tras 
% calcular explícitamente dichas potencias.
%
%
% Datos de entrada:
% - f:          Función a aplicar sobre la matriz ('exp','cos','cosh', ...)
%               o acción de la función ('expv','cosv','coshv', ...).
% - metodo_f:   Método a emplear para calcular f(A) (taylor, bernoulli,
%               hermite, ...).
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz de la cual calculamos f(A).
% - mmin:       Valor mínimo del grado del polinomio de aproximación.
%               Valores posibles son 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 42
%               49, 56, 64.
% - mmax:       Valor máximo del grado del polinomio de aproximación.
%               Valores posibles son 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 42
%               49, 56, 64.
%
% Datos de salida:
% - m:          Orden de la aproximación polinómica a f(A). Coincide con el 
%               grado del polinomio de aproximación a f(A).
% - sm:         Valor del escalado de la matriz.
% - pA:         Vector de celdas con las potencias de A, de modo que pA{i}
%               contiene A^i, para i=1,2,3,...,q, siendo q=ceil(sqrt(m)).
%               Si empleamos la función coseno y sólo trabajamos con los
%               términos pares del polinomio, el vector tendrá las
%               potencias de B, siendo B=A^2.
% - nProd:      Número de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

% Elegimos los tipos de errores que proporcionan mejores resultados

% A PRIORI, LOS RESULTADOS SON MEJORES CON BACKWARD RELATIVOS

switch f
    case {'exp','expv'}
        %tipo_error_1='forward';
        tipo_error_1='backward';
        %tipo_error_2='absoluto';
        tipo_error_2='relativo';
    case 'cos'
        %tipo_error_1='forward';
        tipo_error_1='backward';
        %tipo_error_2='absoluto';
        tipo_error_2='relativo';
    case 'cosh'
        tipo_error_1='forward';
        tipo_error_2='absoluto';        
    otherwise
        error('Función matricial no contemplada');
end

% Obtenemos los valores de theta y M
[theta,M]=get_theta(f,metodo_f,tipo_error_1,tipo_error_2);

% Buscamos las posiciones (imin e imax) de mmin y mmax en el vector M
if mmin<M(1)
    mmin=M(1);
elseif mmin>M(end)
    mmin=M(end);
end

if mmax<M(1)
    mmax=M(1);
elseif mmin>M(end)
    mmax=M(end);
end

if mmin>mmax
    error('Valor mmin mayor que mmin');
end 

i=1;
encontrado=0;
while i<=length(M) && encontrado==0
    if M(i)==mmin
        imin=i;
        encontrado=1;
    else
        i=i+1;
    end
end
if (encontrado==0)
    error('Valor mmin no permitido (emplear 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 42, 49, 56 o 64)');
end

i=1;
encontrado=0;
while i<=length(M) && encontrado==0
    if M(i)==mmax
        imax=i;
        encontrado=1;
    else
        i=i+1;
    end
end
if (encontrado==0)
    error('Valor mmax no permitido (emplear 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 42, 49, 56 o 64)');
end

% En caso de trabajar únicamente con los términos pares del polinomio, 
% obtenemos potencias de B=A^2
switch f
    case 'exp'
        factor_s=1;
    case 'cos'
        switch metodo_f
            case 'taylor'
                A=A*A;
                factor_s=0.5;
            case 'bernoulli'
                formulacion=get_formulacion_cos_bernoulli;
                switch formulacion
                    case 'terminos_pares_polinomio_solo_pares'
                        A=A*A;
                        factor_s=0.5;
                    otherwise
                        factor_s=1;
                end
        end
end
% El vector c estará formado por los coeficientes de mayor orden de los polinomios de orden m=2,
% 4, 6, 9, 12, 16, 20, 25, 30, 36, 42, 49, 56, 64.
switch f
    case {'exp','expv'}
        switch metodo_f
            case {'taylor','taylor_bernoulli'}
                %c=[1/6 1/120 1/5040 1/3628800 1/6227020800 1/355687428096000 1/51090942171709440000 1/403291461126605635584000000 1/8222838654177922817725562880000000 1/13763753091226345046315979581580902400000000 1/60415263063373835637355132068513997507264512000000000 1/30414093201713378043612608166064768844377641568960512000000000000 1/40526919504877216755680601905432322134980384796226602145184481280000000000000 1/8247650592082470666723170306785496252186258551345437492922123134388955774976000000000000000];
                c=[1/2 1/24 1/720 1/362880 1/479001600 1/20922789888000 1/2432902008176640000 1/15511210043330986055303168 1/265252859812191032188804700045312 1/371993326789901177492420297158468206329856 1/1405006117752879788779635797590784832178972610527232 1/608281864034267522488601608116731623168777542102418391010639872 1/710998587804863481025438135085696633485732409534385895375283304551881375744 1/126886932185884165437806897585122925290119029064705209778508545103477590511637067401789440];
            case 'bernoulli'
                c=[1934613350591413/2251799813685248 1934613350591413/27021597764222976 1934613350591413/810647932926689280 1934613350591413/408566558195051397120 1934613350591413/539307856817467844198400 1934613350591413/23556967185786995434586112000 1934613350591413/2739204144363311829133673103360000 1934613350591413/17464069942802730978104876446565984632832 1934613350591413/298648170152285486541805842340434348111652978688 1934613350591413/418827251978827522481019018066584581083630589284024582144 1934613350591413/1581896257091284160381694378362351503325668515799159736213090336768 1934613350591413/684864494050239481748780073519512412990706924028991916448058871191601443504128 1934613350591413/800513243774733013679858865737379924724552612330200369222259688023035332288044772418912256 1934613350591413/142861985127633330739513693855108127853356657352424120353951214436608163739431367321068067010158065090560];
        end
    case 'cos'
         switch metodo_f
            case 'taylor'
                c=[1/24 1/40320 1/479001600 -1/6402373705728000 1/620448401733239409999872 1/263130836933693517766352317727113216 1/815915283247897683795548521301193790359984930816 -1/30414093201713375576366966406747986832057064836514787179557289984 1/8320987112741391580056396102959641077457945541076708813599085350531187384917164032 1/61234458376886076682034243918084408426143679367126656631657903381829221022872956916891969827292894461952 1/3314240134565351991893962785187002255986138585985099085000359647021178112607661449751964466234594461331925608329126314254532608 -1/9426890448883242029410148360874034376137592466180063696357188182614769297217373775790759257383412737431326439501183709769874985637770333212700442263289856 1/197450685722107283218203224975563190350604098858598125302874203564961210901795886052940088784315404959246758100114554585320567376838138578357747136064291560700269291680195376450633728 1/385620482362580254065032983367256046768928224415294665452474122816236575174870722402856406966606225777437456487212434754604702120071219780114893563585389807556408608834254609008155210269502377566782820110682950729728];
            case 'bernoulli'
                c=[];
        end       
end

pA{1}=A;
fin=0;
im=0;
nProd=0;
while fin==0 && im<imax
    im=im+1;
    j=ceil(sqrt(M(im)));
    if sqrt(M(im))>floor(sqrt(M(im)))
        switch plataforma
            case 'sinGPUs'
                pA{j}=pA{j-1}*A;
            case 'conGPUs'
                pA{j}=call_gpu('power');
        end
        nProd=nProd+1;        
    end
    a(im)=norm1p(A,M(im));
    alfa(im)=a(im)^(1/M(im));
    % if alfa(im)<theta(im)
    if alfa(im)<theta(im) && im>=imin    
        fin=1;     
    end
end

if fin==1
    sm=0;
else
    sm=ceil(max(0,factor_s*log2(alfa(im)/theta(im))));
    if c(im)*a(im)/2^((sm-1)*M(im))<eps/2
        sm=sm-1;
        if c(im)*a(im)/2^((sm-1)*M(im))<eps/2 && sm>0
            sm=sm-1;
        end
    end    
end  
m=M(im);
end
