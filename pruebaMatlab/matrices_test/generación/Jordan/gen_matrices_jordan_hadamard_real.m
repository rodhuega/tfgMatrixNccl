function gen_matrices_jordan_hadamard_real(F,n,bound_vp,max_mult,nmat,nd)
% gen_matrices_jordan_hadamard_real(F,n,bound_vp,max_mult,nmat,nd)
%
% Genera matrices de test de Jordan a partir de matrices V de Hadamard con 
% valores propios reales comprendidos en un intervalo concreto y con una
% multiplicidad determinada. Por cada matriz se genera un fichero.mat que 
% almacena:
% - A:          La propia matriz, generada como V*D*V'. 
% - normA:      1-norma de la matriz A.
% - condV:      Número de condición de la matriz V (vectores propios).
% - condFA:     Número de condición de la función matricial.
% - FA:         Resultado de la función matricial aplicada sobre la matriz A.
% - flag_error: Indica si ha ocurrido algún error. Valdrá 0 (no hay error),
%               -2 (ha ocurrido un error en el cálculo de la matriz A y 
%               A valdrá infinito o not a number), -3 (ha ocurrido un error 
%               en el cálculo de la función matricial y FA valdrá infinito 
%               o not a number). 
% Adicionalmente, se crea un fichero .m con el número de matrices
% generadas.
%
% Datos de entrada:
% - F:        Función a aplicar sobre la matriz (@exp, @cos, @cosh, ...).
% - n:        Número de filas y columnas de la matriz. Para ver los tamaños
%             posibles, consultar las restricciones de la generación de las
%             matrices de Hadamard (n, n/12 or n/20 deben ser potencias de 
%             2).
% - bound_vp: Valor que delimina el intervalo ]-bound_vp, bound_vp[ en el 
%             que se encuentran los valores propios de la matriz.
% - max_mult: Determina el intervalo [1, max_mult] en el que se encuentran
%             las multiplicidades de los valores propios.
% - nmat:     Número de matrices a generar.
% - nd:       Número de dígitos vpa (32, 64, 128, 256).
%
% Ejemplo de invocación:
% gen_matrices_jordan_hadamard_real(@exp,128,10,5,100,256)

digits(nd);
t0=sprintf('%s_jordan_hadamard_real_n%d_boundvp%d_maxmult%d_nd%d',func2str(F),n,bound_vp,max_mult,nd);
try
    warning off
    eval(sprintf('mkdir %s',t0));
catch
end
rng('default');
for k=1:nmat
    tic
    [A,FA,condV]=jordan_hadamard_real(F,n,bound_vp,max_mult,nd);
    normA=norm(A,1);
    condFA=funm_condest1(A,F);
	if max(max((isnan(A)))) || max(max((isinf(A))))
        flag_error=-2;        
        fprintf('La matriz %d no se ha podido calcular (A=Inf o A=NaN)\n',k);
    elseif max(max((isnan(FA))))||max(max((isinf(FA))))
        flag_error=-3;
        fprintf('La función %s no se ha podido aplicar sobre la matriz %d (FA=Inf o FA=NaN)\n',func2str(F),k);
    else
        flag_error=0;
    end
	t=sprintf('save %s/%s_%d.mat A FA normA condFA condV flag_error',t0,t0,k);
	eval(t);
    fprintf('Matriz %d (norm(A,1)=%f, cond(V)=%f): %f segundos\n',k,normA,condV,toc)
end
t=sprintf('%s/%s.m',t0,t0);
fileID = fopen(t,'w');
fprintf(fileID,'nmat=%d;',nmat);
fclose(fileID);
end