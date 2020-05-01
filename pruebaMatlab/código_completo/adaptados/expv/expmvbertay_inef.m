function [fAv,m,s,np] = expmvbertay_inef(A,v,seleccion_ms)
% [fA,m,s,np] = expmvtay(A,v,seleccion_ms)
%
% C�lculo de la accion exponencial matricial por Taylor.
%
% Datos de entrada:
% - A:            Matriz de la cual calculamos f(A).
% - v:            Vector para calcular f(A)*v.
% - seleccion_ms: Modo de calcular el grado del polinomio (m) y el 
%                 escalado (s). Toma los siguientes valores:
%   * 1:          C�lculo mediante 'conEstNorma'.
%   * 2:          C�lculo mediante 'conEstNormaSolo'.
%   * 3:          C�lculo mediante 'conEstNormaPotencias'.
%
% Datos de salida:
% - fAv:          Valor de la acci�n f(A)*v.
% - m:            Grado del polinomio empleado en la aproximaci�n.
% - s:            Valor del escalado de la matriz.
% - nProd:        N�mero de productos matriciales involucrados para calcular
%                 f(A).

switch seleccion_ms
    case 1
        [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','conEstNorma','sinGPUs',A);
    case 2
        [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','conEstNormaSolo','sinGPUs',A);
    case 3
        [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','conEstNormaPotencias','sinGPUs',A);
end
fAv=fA*v;