function [m,sm,pA,nProd]=selecciona_ms_conEstNormaConPotencias(f,metodo_f,plataforma,A,mmin,mmax)
% [m,sm,pA,nProd]=selecciona_ms_conEstNormaConPotencias(f,metodo_f,plataforma,A,mmin,mmax)
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
                    case {'terminos_pares_polinomio_completo','terminos_pares_impares_polinomio_completo'}
                        factor_s=1;
                        fin_s_m=1;
                end
        end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cálculo de las potencias previas a imin
pA{1}=A;
nProd=0;
for im=1:imin-1
    j=pot(im);
    if sqrt(M(im))>floor(sqrt(M(im)))
        switch plataforma
            case 'sinGPUs'
                pA{j}=pA{j-1}*A;
            case 'conGPUs'
                pA{j}=call_gpu('power');
        end
        nProd=nProd+1;        
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cálculo de las potencias entre imin e imax
fin=0;
im=imin;
while fin==0 && im<=imax
    j=pot(im);
    if sqrt(M(im))>floor(sqrt(M(im)))
        switch plataforma
            case 'sinGPUs'
                pA{j}=pA{j-1}*A;
            case 'conGPUs'
                pA{j}=call_gpu('power');
        end
        nProd=nProd+1;        
    end
    switch tipo_error_2
        case 'absoluto' % Error absoluto
            alfa(im)=norm1pp(pA{j},j,A)^(1/(M(im)+1));
        case 'relativo' % Error relativo
            alfa(im)=norm1p(pA{j},j)^(1/(M(im)));
    end
    if alfa(im)<theta(im)
        fin=1;
    else 
        im=im+1;
    end
end

if fin==1
    sm=0;
    q_inic=pot(im);
else
    im=imax; % se habrá salido en el bucle anterior porque im>imax
    q_inic=pot(im);
    sm=ceil(max(0,factor_s*log2(alfa(im)/theta(im))));
    j=im;
    fin=fin_s_m;
    while fin==0 && j>imin
        j=j-1;
        s=ceil(max(0,factor_s*log2(alfa(j)/theta(j))));
        if sm>=s
            sm=s;
            im=j;
        else
            fin=1;
        end
    end
end  
m=M(im);
q_final=pot(im);
% Es posible que se hayan evaluado más potencias de las necesarias. Si es 
% así, liberamos la memoria reservada por exceso.
if q_inic>q_final
    switch plataforma
        case 'sinGPUs'
            pA(q_final+1:q_inic)=[];
        case 'conGPUs'
            call_gpu('free',q_final-q_inic);
    end
end
end
