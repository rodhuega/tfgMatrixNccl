function [m,sm,pA,nProd]=selecciona_ms_conEstNormaSinPotencias(f,metodo_f,plataforma,A,mmin,mmax)
% [m,sm,pA,nProd]=selecciona_ms_conEstNormaSinPotencias(f,metodo_f,plataforma,A,mmin,mmax)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz estimando la norma de las potencias de la matriz sin 
% calcular explícitamente dichas potencias.
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
pot=ceil(sqrt(M));

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
        fin_s_m=1;
    case 'cos'
        switch metodo_f
            case 'taylor'
                A=A*A;
                factor_s=0.5;
                fin_s_m=0;
            case 'bernoulli'
                formulacion=get_formulacion_cos_bernoulli;
                switch formulacion
                    case 'terminos_pares_polinomio_solo_pares'
                        A=A*A;
                        factor_s=0.5;
                        fin_s_m=0;
                    otherwise
                        factor_s=1;
                        fin_s_m=1;
                end
        end
end
   
tol=1e-2;
alpham=compute_alpha(A,tol,imax);
fin=0;
im=imin; % comenzamos en imin
while fin==0 && im<=imax
    if alpham<theta(im)
        fin=1;
    else
        im=im+1;
    end
end
if fin==1
    sm=0;
else
    im=imax; % se habrá salido en el bucle anterior porque im>imax
    sm=ceil(max(0,factor_s*log2(alpham/theta(im))));
    j=im;
    fin=fin_s_m;    
    while fin==0
        j=j-1;
        s=ceil(max(0,factor_s*log2(alpham/theta(j))));
        %if sm>=s
        if sm>=s && j>=imin
            sm=s;
            im=j;
        else
            fin=1;
        end
    end
end 

m=M(im);

% Cálculo de las potencias de A
q=pot(im);
nProd=0;
switch plataforma
    case 'sinGPUs'
        pA = cell(q,1);
        pA{1}=A;
        for i=2:q
            pA{i}=pA{i-1}*A;
            nProd=nProd+1;
        end
    case 'conGPUs'
        pA=[];
        for i=2:q
            call_gpu('power');
            nProd=nProd+1;
        end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [alpha,i]=compute_alpha(A,tol,imax)
M= [2 4 6 9 12 16 20 25 30 36 42 49 56 64];
fin=0;
i=1;
p=M(i)+1;
alpha0=norm1p(A,p);
while fin==0 && i<imax
    i=i+1;
    p=M(i)+1;
    alpha=norm1p(A,p)^(1/p);
    if (alpha-alpha0)/alpha>tol
        alpha=alpha0;
    else
        fin=1;
    end
end
end



