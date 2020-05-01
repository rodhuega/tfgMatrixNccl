function [fA,m,s,np] = expmber_gpu(A,seleccion_ms)
% [fA,m,s,np] = expmber_gpu(A,seleccion_ms)
%
% Cálculo de una exponencial matricial por Bernoulli mediante GPUs.
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
        [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNorma','conGPUs',A);
    case 2
        [fA,m,s,np] = fun_pol('exp','bernoulli','sinEstNormaSplines','conGPUs',A);
    case 3
        [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaSplines','conGPUs',A);        
    case 4
        [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaSinPotencias','conGPUs',A);
    case 5
        [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaConPotencias','conGPUs',A);
    case 6
        [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaConPotenciasNuevo','conGPUs',A);        
end
end