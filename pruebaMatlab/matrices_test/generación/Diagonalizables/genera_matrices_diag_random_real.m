function genera_matrices_diag_random_real(F,n,maxv,nmat,nd)
% genera_matrices_diag_random_real(F,n,maxv,nmat,nd)
%
% Genera matrices de test diagonalizables a partir de matrices V de n�meros 
% aleatorios, con valores propios reales de magnitud creciente. Por cada 
% matriz se genera un fichero .mat que almacena:
% - A:          La propia matriz, generada como V^-1*D*V. 
% - normA:      1-norma de la matriz A.
% - condV:      N�mero de condici�n de la matriz V (vectores propios).
% - condFA:     N�mero de condici�n de la funci�n matricial.
% - FA:         Resultado de la funci�n matricial aplicada sobre la matriz A.
% - flag_error: Indica si ha ocurrido alg�n error. Valdr� 0 (no hay error),
%               -2 (ha ocurrido un error en el c�lculo de la matriz A y 
%               A valdr� infinito o not a number), -3 (ha ocurrido un error 
%               en el c�lculo de la funci�n matricial y FA valdr� infinito 
%               o not a number). 
% Adicionalmente, se crea un fichero .m con el n�mero de matrices
% generadas.
%
% Datos de entrada:
% - F:    Funci�n a aplicar sobre la matriz (@exp, @cos, @cosh, ...).
% - n:    N�mero de filas y columnas de la matriz. 
% - maxv: Valor que determina el intervalo [maxv, nmat*maxv] en el que se 
%         encuentra la 1-norma de las matrices generadas.
% - nmat: N�mero de matrices a generar.
% - nd:   N�mero de d�gitos vpa (32, 64, 128, 256).
%
% Ejemplo de invocaci�n:
% genera_matrices_diag_random_real(@exp,128,5,100,256)

digits(nd);
t0=sprintf('%s_diag_random_real_n%d_maxv%d_nd%d',func2str(F),n,maxv,nd);
try
    warning off
    eval(sprintf('mkdir %s',t0));
catch
end

for i=1:nmat
    tic
    D=rand(n,1)-0.5;
    V=vpa(rand(n));
    Vd=double(V);
    condV=cond(double(V));
    nA=vpa(norm(Vd*diag(D)*Vd',1));
    D=vpa(vpa(D)*i*maxv/nA);
    A=double(vpa(vpa(V\diag(D))*V));
    normA=norm(A,1);
    condFA=funm_condest1(A,F);
    fD=vpa(F(D));
	FA=double(vpa(vpa(V\diag(fD))*V));
	if max(max((isnan(A)))) || max(max((isinf(A))))
        flag_error=-2;        
        fprintf('La matriz %d no se ha podido calcular (A=Inf o A=NaN)\n',i);
    elseif max(max((isnan(FA))))||max(max((isinf(FA))))
        flag_error=-3;
        fprintf('La funci�n %s no se ha podido aplicar sobre la matriz %d (FA=Inf o FA=NaN)\n',func2str(F),i);
    else
        flag_error=0;
    end
	t=sprintf('save %s/%s_%d.mat A FA normA condFA condV flag_error',t0,t0,i);
	eval(t);
	fprintf('Matriz %d (norm(A,1)=%f, cond(V)=%f): %f segundos\n',i,normA,condV,toc)
end
t=sprintf('%s/%s.m',t0,t0);
fileID = fopen(t,'w');
fprintf(fileID,'nmat=%d;',nmat);
fclose(fileID);
end

