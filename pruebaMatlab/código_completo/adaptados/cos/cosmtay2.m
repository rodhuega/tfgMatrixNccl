function [fA,m,s,np] = cosmtay2(A,seleccion_ms)
% [fA,m,s,np] = cosmtay2(A,seleccion_ms)
%
% C�lculo del coseno matricial por Taylor.
%
% Datos de entrada:
% - A:     Matriz de la cual calculamos f(A).
% - seleccion_ms: Modo de calcular el grado del polinomio (m) y el 
%          escalado (s). Toma los siguientes valores:
%   * 1:   C�lculo mediante 'sinEstNorma'.
%   * 2:   C�lculo mediante 'conEstNorma'.
%   * 3:   C�lculo mediante 'conEstNormaSinPotencias'.
%   * 4:   C�lculo mediante 'conEstNormaConPotencias'.
%   * 5:   C�lculo mediante 'conEstNormaConPotenciasNuevo'.
%
% Datos de salida:
% - fA:    Valor de la funci�n f sobre la matriz A.
% - m:     Grado del polinomio empleado en la aproximaci�n.
% - s:     Valor del escalado de la matriz.
% - nProd: N�mero de productos matriciales involucrados para calcular f(A).

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