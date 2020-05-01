function [fA,m,s,np] = expmber(A,seleccion_ms)
% [fA,m,s,np] = expmber(A,seleccion_ms)
%
% C�lculo de una exponencial matricial por Bernoulli.
%
% Datos de entrada:
% - A:            Matriz de la cual calculamos f(A).
% - seleccion_ms: Modo de calcular el grado del polinomio (m) y el 
%                 escalado (s). Toma los siguientes valores:
%   * 1:          C�lculo mediante 'conEstNorma'.
%   * 2:          C�lculo mediante 'sinEstNormaSplines'.
%   * 3:          C�lculo mediante 'conEstNormaSplines'.
%   * 4:          C�lculo mediante 'conEstNormaSinPotencias'.
%   * 5:          C�lculo mediante 'conEstNormaConPotencias'.
%   * 6:          C�lculo mediante 'conEstNormaConPotenciasNuevo'.
%
% Datos de salida:
% - fA:           Valor de la funci�n f sobre la matriz A.
% - m:            Grado del polinomio empleado en la aproximaci�n.
% - s:            Valor del escalado de la matriz.
% - nProd:        N�mero de productos matriciales involucrados para calcular
%                 f(A).

switch seleccion_ms
    case 1
        [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNorma','sinGPUs',A);
    case 2
        [fA,m,s,np] = fun_pol('exp','bernoulli','sinEstNormaSplines','sinGPUs',A);
    case 3
        [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaSplines','sinGPUs',A);        
    case 4
        [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaSinPotencias','sinGPUs',A);
    case 5
        [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaConPotencias','sinGPUs',A);
    case 6
        [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaConPotenciasNuevo','sinGPUs',A);        
end
end