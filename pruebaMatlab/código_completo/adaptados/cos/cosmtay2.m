function [fA,m,s,np] = cosmtay2(A,seleccion_ms)
% [fA,m,s,np] = cosmtay2(A,seleccion_ms)
%
% Cálculo del coseno matricial por Taylor.
%
% Datos de entrada:
% - A:     Matriz de la cual calculamos f(A).
% - seleccion_ms: Modo de calcular el grado del polinomio (m) y el 
%          escalado (s). Toma los siguientes valores:
%   * 1:   Cálculo mediante 'sinEstNorma'.
%   * 2:   Cálculo mediante 'conEstNorma'.
%   * 3:   Cálculo mediante 'conEstNormaSinPotencias'.
%   * 4:   Cálculo mediante 'conEstNormaConPotencias'.
%   * 5:   Cálculo mediante 'conEstNormaConPotenciasNuevo'.
%
% Datos de salida:
% - fA:    Valor de la función f sobre la matriz A.
% - m:     Grado del polinomio empleado en la aproximación.
% - s:     Valor del escalado de la matriz.
% - nProd: Número de productos matriciales involucrados para calcular f(A).

switch seleccion_ms
    case 1
        [fA,m,s,np] = fun_pol('cos','taylor','sinEstNorma','sinGPUs',A);
    case 2
        [fA,m,s,np] = fun_pol('cos','taylor','conEstNorma','sinGPUs',A);
    case 3
        [fA,m,s,np] = fun_pol('cos','taylor','conEstNormaSinPotencias','sinGPUs',A);
    case 4
        [fA,m,s,np] = fun_pol('cos','taylor','conEstNormaConPotencias','sinGPUs',A);  
    case 5
        [fA,m,s,np] = fun_pol('cos','taylor','conEstNormaConPotenciasNuevo','sinGPUs',A);  
        
end
end