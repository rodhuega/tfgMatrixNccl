function [fA,nProd] = calcula_fun_pol(f,metodo_f,plataforma,pA,v,m,s)
% [fA,nProd] = calcula_fun_pol(f,metodo_f,plataforma,pA,v,m,s)
% C�lculo de funciones de matrices, o de sus acciones, a partir de 
% aproximaciones para unos valores de m y s concretos.
%
% Datos de entrada:
% - f:          Funci�n a aplicar sobre la matriz ('exp','cos','cosh', ...)
%               o acci�n de la funci�n ('expv','cosv','coshv', ...).
% - metodo_f:   M�todo a emplear para calcular f(A) (taylor, bernoulli,
%               hermite, ...).
% - plataforma: Decide si calculamos la funci�n matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - pA:         Vector de cells arrays con las potencias de A, de modo que
%               pA{i} contiene:
%               a) A^i, para i=1,2,3,...,q, siendo q=ceil(sqrt(m)) o 
%                  q=floor(sqrt(m)), en caso de la funci�n exponencial.
%               b) B^i, para i=1,2,3,...,q, siendo q<=4 y B=A*A, en caso
%                  de la funci�n coseno o coseno hiperb�lico.
%               En caso de querer calcular la acci�n, valdr� A, en forma
%               de cell array de una �nica componente.
% - v:          Vector columna, de dimensi�n igual a A, para calcular el 
%               valor de la acci�n de la funci�n matricial, es decir, 
%               f(A)*v sin obtener f(A) de forma expl�cita. En caso de no 
%               querer aplica dicha acci�n, el vector estar� vac�o.
% - m:          Orden de la aproximaci�n polin�mica a f(A).
% - s:          Valor del escalado de la matriz.
%
% Datos de salida:
% - fA:         Valor de la funci�n f sobre la matriz A o valor de la 
%               acci�n de la funci�n (f(A)*v).
% - nProd:      N�mero de productos matriciales necesarios para calcular
%               f(A).

switch f
    % Funci�n matricial
    case 'exp'
        [fA,nProd]=exp_pol(metodo_f,plataforma,pA,m,s);
    case 'cos'
        [fA,nProd]=cos_pol(metodo_f,plataforma,pA,m,s);
    case 'cosh'
        [fA,nProd]=cosh_pol(metodo_f,plataforma,pA,m,s);
    % Acci�n de la funci�n matricial (f(A)*v)
    case 'expv'
        fA=expv_pol(metodo_f,plataforma,pA,v,m,s);    
        nProd=0;
    otherwise
        error('Funci�n matricial o acci�n no contemplada\n');
end
    
end


