function [fA,m,s,np] = expmtay(A,seleccion_ms)
% [fA,m,s,np] = expmtay(A,seleccion_ms)
%
% Cálculo de una exponencial matricial por Taylor.
%
% Datos de entrada:
% - A:            Matriz de la cual calculamos f(A).
% - seleccion_ms: Modo de calcular el grado del polinomio (m) y el 
%                 escalado (s). Toma los siguientes valores:
%   * 1:          Cálculo mediante 'conEstNorma'.
%   * 2:          Cálculo mediante 'sinEstNormaSplines'.
%   * 3:          Cálculo mediante 'conEstNormaSplines'.
%   * 4:          Cálculo mediante 'conEstNormaSinPotencias'.
%   * 5:          Cálculo mediante 'conEstNormaConPotencias'.
%   * 6:          Cálculo mediante 'conEstNormaConPotenciasNuevo'.
%
% Datos de salida:
% - fA:           Valor de la función f sobre la matriz A.
% - m:            Grado del polinomio empleado en la aproximación.
% - s:            Valor del escalado de la matriz.
% - nProd:        Número de productos matriciales involucrados para calcular
%                 f(A).

switch seleccion_ms
    case 1
        [fA,m,s,np] = fun_pol('exp','taylor','conEstNorma','sinGPUs',A);
    case 2
        [fA,m,s,np] = fun_pol('exp','taylor','sinEstNormaSplines','sinGPUs',A);
    case 3
        [fA,m,s,np] = fun_pol('exp','taylor','conEstNormaSplines','sinGPUs',A);
    case 4
        [fA,m,s,np] = fun_pol('exp','taylor','conEstNormaSinPotencias','sinGPUs',A);
    case 5
        [fA,m,s,np] = fun_pol('exp','taylor','conEstNormaConPotencias','sinGPUs',A);
    case 6
        [fA,m,s,np] = fun_pol('exp','taylor','conEstNormaConPotenciasNuevo','sinGPUs',A);
end
end