function [fA,m,s,np] = cosmber(A,formulacion_seleccion_ms)
% [fA,m,s,np] = cosmber(A,formulacion_seleccion_ms)
%
% C�lculo del coseno matricial por Bernoulli.
%
% Datos de entrada:
% - A:     Matriz de la cual calculamos f(A).
% - formulacion_seleccion_ms: Vector de 2 componentes. El primero de ellos
%          almacena la formulaci�n te�rica a emplear. Toma estos valores:
%   * 1:   C�lculo mediante 'terminos_pares_polinomio_completo'.
%   * 2:   C�lculo mediante 'terminos_pares_impares_polinomio_completo'.
%   * 3:   C�lculo mediante 'terminos_pares_polinomio_solo_pares'.
%          El segundo elemento indica el modo de calcular el grado del 
%          polinomio (m) y el escalado (s). Toma los siguientes valores:
%   * 1:   C�lculo mediante 'sinEstNorma'.
%   * 2:   C�lculo mediante 'conEstNorma'.
%   * 3:   C�lculo mediante 'conEstNormaSinPotencias'.
%   * 4:   C�lculo mediante 'conEstNormaConPotencias'.
%   * 5:   C�lculo mediante 'conEstNormaConPotenciasNuevo'.
%
%   Aclaraci�n en la invocaci�n: 
%   * Formulaci�n te�rica 1 y 2: c�lculo de m y s mediante las opciones de
%                                la 3 a la 5.
%   * Formulaci�n te�rica 3:     c�lculo de m y s mediante las opciones de
%                                la 1 a la 5.
%
% Datos de salida:
% - fA:    Valor de la funci�n f sobre la matriz A.
% - m:     Grado del polinomio empleado en la aproximaci�n.
% - s:     Valor del escalado de la matriz.
% - nProd: N�mero de productos matriciales involucrados para calcular f(A).

% Ejemplos de invocaci�n:
% - [fA,m,s,np] = cosmber(A,[[1/2] [3/4/5]])
% - [fA,m,s,np] = cosmber(A,[3 [1/2/3/4/5]])

%plataforma='sinGPUs';
plataforma='conGPUs';

if formulacion_seleccion_ms(1)<1 || formulacion_seleccion_ms(1)>3
    error('Formulaci�n te�rica no contemplada');
elseif formulacion_seleccion_ms(2)<1 || formulacion_seleccion_ms(2)>5
    error('Formulaci�n de c�lculo de m y s no contemplada');
end

switch formulacion_seleccion_ms(1)
    case {1,2}
        switch formulacion_seleccion_ms(2)
            case {1,2}
                error('Combinaci�n no contemplada');
        end
end   

switch formulacion_seleccion_ms(1)
    case 1
        formulacion='terminos_pares_polinomio_completo';
    case 2
        formulacion='terminos_pares_impares_polinomio_completo';       
    case 3
        formulacion='terminos_pares_polinomio_solo_pares';
end
set_formulacion_cos_bernoulli(formulacion);

switch formulacion_seleccion_ms(2)
    case 1
        [fA,m,s,np] = fun_pol('cos','bernoulli','sinEstNorma',plataforma,A);
    case 2
        [fA,m,s,np] = fun_pol('cos','bernoulli','conEstNorma',plataforma,A);
    case 3
        [fA,m,s,np] = fun_pol('cos','bernoulli','conEstNormaSinPotencias',plataforma,A);
    case 4
        [fA,m,s,np] = fun_pol('cos','bernoulli','conEstNormaConPotencias',plataforma,A);  
    case 5
        [fA,m,s,np] = fun_pol('cos','bernoulli','conEstNormaConPotenciasNuevo',plataforma,A);
end
end
