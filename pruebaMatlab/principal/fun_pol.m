function [fA,m,s,nProd] = fun_pol(f,metodo_f,metodo_ms,plataforma,A,v)
% [fA,m,s,nProd] = fun_pol(f,metodo_f,metodo_ms,plataforma,A,v)
%
% Funci�n principal para el c�lculo de funciones de matrices, a partir de 
% aproximaciones, y de las acciones de dichas funciones.
%
% Datos de entrada:
% - f:          Funci�n a aplicar sobre la matriz ('exp','cos','cosh', ...)
%               o acci�n de la funci�n ('expv','cosv','coshv', ...).
% - metodo_f:   M�todo a emplear para calcular f(A) ('taylor', 'bernoulli',
%               'hermite', ...).
% - metodo_ms:  M�todo a usar para calcular el grado del polinomio (m) y 
%               el valor del escalado (s). Ejemplos: 'conEstNorma' (con 
%               estimaciones de normas de las potencias matriciales), 
%               'sinEstNorma' (sin estimaciones de normas de las potencias 
%               matriciales).
% - plataforma: Decide si calculamos la funci�n matricial mediante Matlab
%               ('sinGPUs') o mediante GPUs ('conGPUs').
% - A:          Matriz de la cual calculamos f(A).
% - v:          Vector columna, de dimensi�n igual a A, para calcular el 
%               valor de la acci�n de la funci�n matricial, es decir, 
%               f(A)*v sin obtener f(A) de forma expl�cita. En caso de no 
%               querer aplica dicha acci�n, el vector no se pasar� como 
%               dato de entrada a la funci�n.
%
% Datos de salida:
% - fA:         Valor de la funci�n f sobre la matriz A o valor de la 
%               acci�n de la funci�n.
% - m:          Grado del polinomio empleado en la aproximaci�n.
% - s:          Valor del escalado de la matriz.
% - nProd:      N�mero de productos matriciales involucrados para calcular
%               f(A).
%
% Ejemplos de invocaci�n:
% * Funci�n exponencial:
%   - [fA,m,s,np] = fun_pol('exp','taylor','conEstNorma','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor','conEstNorma','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor','sinEstNormaSplines','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor','sinEstNormaSplines','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor','conEstNormaSplines','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor','conEstNormaSplines','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor','conEstNormaSinPotencias','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor','conEstNormaSinPotencias','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor','conEstNormaConPotencias','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor','conEstNormaConPotencias','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor','conEstNormaConPotenciasNuevo','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor','conEstNormaConPotenciasNuevo','conGPUs',A)
%
%   - [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNorma','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNorma','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','bernoulli','sinEstNormaSplines','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','bernoulli','sinEstNormaSplines','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaSplines','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaSplines','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaSinPotencias','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaSinPotencias','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaConPotencias','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaConPotencias','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaConPotenciasNuevo','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','bernoulli','conEstNormaConPotenciasNuevo','conGPUs',A)
%
%   - [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','conEstNorma','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','conEstNorma','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','sinEstNormaSplines','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','sinEstNormaSplines','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','conEstNormaSplines','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','conEstNormaSplines','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','conEstNormaSinPotencias','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','conEstNormaSinPotencias','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','conEstNormaConPotencias','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','conEstNormaConPotencias','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','conEstNormaConPotenciasNuevo','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor_bernoulli','conEstNormaConPotenciasNuevo','conGPUs',A)
%
%   - [fA,m,s,np] = fun_pol('exp','taylor_sastre','sinEstNorma','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','taylor_sastre','conEstNorma','sinGPUs',A)
%
%   - [fA,m,s,np] = fun_pol('exp','splines','sinEstNormaSplines','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('exp','splines','conEstNormaSplines','sinGPUs',A)
%
% * Funci�n coseno:
%   - [fA,m,s,np] = fun_pol('cos','taylor','sinEstNorma','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','taylor','sinEstNorma','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','taylor','conEstNorma','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','taylor','conEstNorma','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','taylor','conEstNormaSinPotencias','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','taylor','conEstNormaSinPotencias','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','taylor','conEstNormaConPotencias','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','taylor','conEstNormaConPotencias','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','taylor','conEstNormaConPotenciasNuevo','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','taylor','conEstNormaConPotenciasNuevo','conGPUs',A)
%
%   - [fA,m,s,np] = fun_pol('cos','bernoulli','conEstNormaSinPotencias','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','bernoulli','conEstNormaSinPotencias','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','bernoulli','conEstNormaConPotencias','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','bernoulli','conEstNormaConPotencias','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','bernoulli','conEstNormaConPotenciasNuevo','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','bernoulli','conEstNormaConPotenciasNuevo','conGPUs',A)
%
%   - [fA,m,s,np] = fun_pol('cos','taylor_sastre','sinEstNorma','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','taylor_sastre','conEstNorma','sinGPUs',A)
%
%   - [fA,m,s,np] = fun_pol('cos','hermite','sinEstNorma','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cos','hermite','sinEstNorma','conGPUs',A)
%
% * Funci�n coseno hiperb�lico:
%   - [fA,m,s,np] = fun_pol('cosh','taylor','sinEstNorma',A)-> SIN
%   IMPLEMENTAR
%   - [fA,m,s,np] = fun_pol('cosh','taylor','conEstNorma',A) -> SIN
%   IMPLEMENTAR
%   - [fA,m,s,np] = fun_pol('cosh','hermite','sinEstNorma','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cosh','hermite','sinEstNorma','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('cosh','hermite','conEstNorma','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cosh','hermite','conEstNorma','conGPUs',A)
%
%   - [fA,m,s,np] = fun_pol('cosh','bernoulli','conEstNormaSinPotencias','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cosh','bernoulli','conEstNormaSinPotencias','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('cosh','bernoulli','conEstNormaConPotencias','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cosh','bernoulli','conEstNormaConPotencias','conGPUs',A)
%   - [fA,m,s,np] = fun_pol('cosh','bernoulli','conEstNormaConPotenciasNuevo','sinGPUs',A)
%   - [fA,m,s,np] = fun_pol('cosh','bernoulli','conEstNormaConPotenciasNuevo','conGPUs',A)
%
% * Acci�n de la funci�n exponencial:
%   - [fA,m,s,np] = fun_pol('expv','taylor','conEstNorma','sinGPUs',A,v)
%   - [fA,m,s,np] = fun_pol('expv','taylor','conEstNormaSinPotencias','sinGPUs',A,v)
%   - [fA,m,s,np] = fun_pol('expv','taylor','conEstNormaConPotencias','sinGPUs',A,v)
%   - [fA,m,s,np] = fun_pol('expv','taylor','conEstNormaConPotenciasNuevo','sinGPUs',A,v)
%   - [fA,m,s,np] = fun_pol('expv','bernoulli','conEstNorma''sinGPUs',A,v)
%   - [fA,m,s,np] = fun_pol('expv','bernoulli','conEstNormaSinPotencias','sinGPUs',A,v)
%   - [fA,m,s,np] = fun_pol('expv','bernoulli','conEstNormaConPotencias','sinGPUs',A,v)
%   - [fA,m,s,np] = fun_pol('expv','bernoulli','conEstNormaConPotenciasNuevo','sinGPUs',A,v)
%   - [fA,m,s,np] = fun_pol('expv','taylor_bernoulli','conEstNorma''sinGPUs',A,v)
%   - [fA,m,s,np] = fun_pol('expv','taylor_bernoulli','conEstNormaSinPotencias','sinGPUs',A,v)
%   - [fA,m,s,np] = fun_pol('expv','taylor_bernoulli','conEstNormaConPotencias','sinGPUs',A,v)
%   - [fA,m,s,np] = fun_pol('expv','taylor_bernoulli','conEstNormaConPotenciasNuevo','sinGPUs',A,v)

if nargin<6
    v=[];
end

% Comprobaci�n de los par�metros de entrada
comprueba_parametros(f,metodo_f,metodo_ms,plataforma);

% Inicializaci�n de la plataforma
inicializa_plataforma(f,metodo_f,plataforma,A);

% Seleccionamos los valores m�s apropiados de m y s para calcular A.
[m,s,pA,nProd_ms]=selecciona_ms(f,metodo_f,metodo_ms,plataforma,A);

% Calculamos f(A) o f(A)*v
[fA,nProd_fA]=calcula_fun_pol(f,metodo_f,plataforma,pA,v,m,s); 

% N�mero final de productos de matrices
nProd=nProd_ms+nProd_fA;

% Finalizaci�n de la plataforma
fA=finaliza_plataforma(plataforma,fA);
end

