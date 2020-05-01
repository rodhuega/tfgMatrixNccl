function [m,s,pA,nProd]=selecciona_ms(f,metodo_f,metodo_ms,plataforma,A)
% [m,s,pA,nProd]=selecciona_ms(f,metodo_f,metodo_ms,plataforma,A)
%
% Obtiene los valores apropiados del grado del polinomio (m) y el escalado
% (s) de la matriz para aplicar la funci�n f sobre A o para calcular la
% acci�n de la funci�n f sin calcular f(A) de forma explicita.
%
% Datos de entrada:
% - f:          Funci�n a aplicar sobre la matriz ('exp','cos','cosh', ...)
%               o acci�n de la funci�n ('expv','cosv','coshv', ...).
% - metodo_f:   M�todo a emplear para calcular f(A) (taylor, bernoulli,
%               hermite, splines, ...).
% - metodo_ms:  M�todo a usar para calcular el grado del polinomio (m) y 
%               el valor del escalado (s). Ejemplos: 'conEstNorma' (con 
%               estimaciones de normas de las potencias matriciales), 
%               'sinEstNorma' (sin estimaciones de normas de las potencias 
%               matriciales).
% - plataforma: Decide si calculamos la funci�n matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz de la cual calculamos f(A).
%
% Datos de salida:
% - m:          Orden de la aproximaci�n polin�mica a f(A).
% - s:          Valor del escalado de la matriz.
% - pA:         Vector de celdas con las potencias de A, de modo que pA{i}
%               contiene:
%               a) A^i, para i=1,2,3,...,q, siendo q=ceil(sqrt(m)) o 
%                  q=floor(sqrt(m)), en caso de la funci�n exponencial y
%                  cuando empleamos los m�todos de Taylor o Bernoulli.
%                  Con el m�todo de taylor_sastre, q<=5.
%                  Tambi�n ocurrir� as� en el c�lculo del coseno por
%                  Bernoulli.
%               b) B^i, para i=1,2,3,...,q, siendo q<=4 y B=A*A, en caso
%                  de la funci�n coseno o coseno hiperb�lico.
%               En caso de calcular la acci�n de la funci�n matricial, ser�
%               un array de celdas de una �nica componente con la matriz A.
% - nProd:      N�mero de productos matriciales llevados a cabo al calcular  
%               las potencias de A.

switch f
    % Funci�n matricial
    case 'exp'
        [m,s,pA,nProd]=selecciona_ms_exp(metodo_f,metodo_ms,plataforma,A);
    case 'cos'
        [m,s,pA,nProd]=selecciona_ms_cos(metodo_f,metodo_ms,plataforma,A);
    case 'cosh'
        [m,s,pA,nProd]=selecciona_ms_cosh(metodo_f,metodo_ms,plataforma,A);
    % Acci�n de la funci�n matricial (f(A)*v)
     case 'expv'
        [m,s,pA,nProd]=selecciona_ms_expv(metodo_f,metodo_ms,plataforma,A);   
    otherwise
        error('Funci�n matricial o acci�n no contemplada');
end    

end



