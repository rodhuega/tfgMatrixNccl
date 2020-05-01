function [m,s,pA,nProd]=selecciona_ms(f,metodo_f,metodo_ms,plataforma,A)
% [m,s,pA,nProd]=selecciona_ms(f,metodo_f,metodo_ms,plataforma,A)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz para aplicar la función f sobre A o para calcular la
% acción de la función f sin calcular f(A) de forma explicita.
%
% Datos de entrada:
% - f:          Función a aplicar sobre la matriz ('exp','cos','cosh', ...)
%               o acción de la función ('expv','cosv','coshv', ...).
% - metodo_f:   Método a emplear para calcular f(A) (taylor, bernoulli,
%               hermite, splines, ...).
% - metodo_ms:  Método a usar para calcular el grado del polinomio (m) y 
%               el valor del escalado (s). Ejemplos: 'conEstNorma' (con 
%               estimaciones de normas de las potencias matriciales), 
%               'sinEstNorma' (sin estimaciones de normas de las potencias 
%               matriciales).
% - plataforma: Decide si calculamos la función matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz de la cual calculamos f(A).
%
% Datos de salida:
% - m:          Orden de la aproximación polinómica a f(A).
% - s:          Valor del escalado de la matriz.
% - pA:         Vector de celdas con las potencias de A, de modo que pA{i}
%               contiene:
%               a) A^i, para i=1,2,3,...,q, siendo q=ceil(sqrt(m)) o 
%                  q=floor(sqrt(m)), en caso de la función exponencial y
%                  cuando empleamos los métodos de Taylor o Bernoulli.
%                  Con el método de taylor_sastre, q<=5.
%                  También ocurrirá así en el cálculo del coseno por
%                  Bernoulli.
%               b) B^i, para i=1,2,3,...,q, siendo q<=4 y B=A*A, en caso
%                  de la función coseno o coseno hiperbólico.
%               En caso de calcular la acción de la función matricial, será
%               un array de celdas de una única componente con la matriz A.
% - nProd:      Número de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

switch f
    % Función matricial
    case 'exp'
        [m,s,pA,nProd]=selecciona_ms_exp(metodo_f,metodo_ms,plataforma,A);
    case 'cos'
        [m,s,pA,nProd]=selecciona_ms_cos(metodo_f,metodo_ms,plataforma,A);
    case 'cosh'
        [m,s,pA,nProd]=selecciona_ms_cosh(metodo_f,metodo_ms,plataforma,A);
    % Acción de la función matricial (f(A)*v)
     case 'expv'
        [m,s,pA,nProd]=selecciona_ms_expv(metodo_f,metodo_ms,plataforma,A);   
    otherwise
        error('Función matricial o acción no contemplada');
end    

end



