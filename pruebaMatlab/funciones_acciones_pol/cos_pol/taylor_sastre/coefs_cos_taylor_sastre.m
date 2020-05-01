function p = coefs_cos_taylor_sastre(m)
% p = coefs_cos_taylor_sastre(m)
% Coeficientes de la aproximación de Taylor y la evaluación polinómica de 
% Sastre en el cálculo de cos(A).
%
% Datos de entrada:
% - m: Orden de la aproximación, de modo que el grado del polinomio a 
%      emplear será 2*m.
%
% Datos de salida:
% - p: Vector de m+1 componentes con los coeficientes de mayor a menor
%      grado.

switch m
    case 0
        p=1;
    case 1
        p=[-1/2 1];
    case 2
        p=[1/24 -1/2 1];
    case 4
        p=[1/40320 -1/720 1/24 -1/2 1];
    case 8
        p= [2.186201576339059e-07
            -2.623441891606870e-05
            6.257028774393310e-03
            -4.923675742167775e-01
            1.441694411274536e-04
            5.023570505224926e+01
            1/24
            -1/2
            1];
    case 12
        p=[  1.269542268337734e-12
            -3.503936660612145e-10
            1.135275478038335e-07
            -2.027712316612395e-05
            1.647243380001247e-03
            -6.469859264308602e-01
            -4.008589447357360e-05
            9.187724869020796e-03
            -1.432942184841715e+02
            4.555439797286385e-03
            1/24
            -1/2
            1];  
    case 15
        p=[ 6.140022498994532e-17
            -2.670909787062621e-14
            1.438284920333222e-11
            -1.050202496489896e-08
            4.215975785860907e-06
            -1.238347173261219e-03
            -3.234597615453461e-09
            9.292820886910254e-07
            2.466381973203188e-01
            -9.369018510939971e-10
            1/3628800
            -1/40320
            1/720
            % De aquí en adelante los añado para que length(p) sea m+1
            1/24
            -1/2 
            1];
 end
end


