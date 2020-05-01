function [fA,nProd] = calcula_fun_pol(f,metodo_f,plataforma,pA,v,m,s)
% [fA,nProd] = calcula_fun_pol(f,metodo_f,plataforma,pA,v,m,s)
% Cálculo de funciones de matrices, o de sus acciones, a partir de 
% aproximaciones para unos valores de m y s concretos.
%
% Datos de entrada:
% - f:          Función a aplicar sobre la matriz ('exp','cos','cosh', ...)
%               o acción de la función ('expv','cosv','coshv', ...).
% - metodo_f:   Método a emplear para calcular f(A) (taylor, bernoulli,
%               hermite, ...).
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - pA:         Vector de cells arrays con las potencias de A, de modo que
%               pA{i} contiene:
%               a) A^i, para i=1,2,3,...,q, siendo q=ceil(sqrt(m)) o 
%                  q=floor(sqrt(m)), en caso de la función exponencial.
%               b) B^i, para i=1,2,3,...,q, siendo q<=4 y B=A*A, en caso
%                  de la función coseno o coseno hiperbólico.
%               En caso de querer calcular la acción, valdrá A, en forma
%               de cell array de una única componente.
% - v:          Vector columna, de dimensión igual a A, para calcular el 
%               valor de la acción de la función matricial, es decir, 
%               f(A)*v sin obtener f(A) de forma explícita. En caso de no 
%               querer aplica dicha acción, el vector estará vacío.
% - m:          Orden de la aproximación polinómica a f(A).
% - s:          Valor del escalado de la matriz.
%
% Datos de salida:
% - fA:         Valor de la función f sobre la matriz A o valor de la 
%               acción de la función (f(A)*v).
% - nProd:      Número de productos matriciales necesarios para calcular
%               f(A).

switch f
    % Función matricial
    case 'exp'
        [fA,nProd]=exp_pol(metodo_f,plataforma,pA,m,s);
    case 'cos'
        [fA,nProd]=cos_pol(metodo_f,plataforma,pA,m,s);
    case 'cosh'
        [fA,nProd]=cosh_pol(metodo_f,plataforma,pA,m,s);
    % Acción de la función matricial (f(A)*v)
    case 'expv'
        fA=expv_pol(metodo_f,plataforma,pA,v,m,s);    
        nProd=0;
    otherwise
        error('Función matricial o acción no contemplada\n');
end
    
end


