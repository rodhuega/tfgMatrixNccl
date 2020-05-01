function [A,fA,condfA,norm1A,e]=fun_eig_vpa(f,A,vmax,nd)
digits(nd);
e=0; 
if vmax~=0
    normA=norm(A,1);
    if normA>vmax
        A=A/normA*vmax;
    end
end
if max(max(isnan(A)))||max(max(isinf(A)))
	fA=[];
    condfA=[];
    norm1A=[];
    e=1;
    return
end
[V,D] = eig(A);
if isinf(cond(V))
    fA=[];
    condfA=[];
    norm1A=[];
    e=1;
    return
end
V=vpa(V);D=vpa(diag(D));
if max(isinf(f(D)))
    fA=[];
    condfA=[];
    norm1A=[];
    e=1;
    return
end
fA=V*diag(vpa(f(D)))/V;
fA=double(fA);
condfA= funm_condest1(A,f);
norm1A=norm(A,1);
if isnan(condfA)||isinf(condfA) %%No se puede calcular el número de condición
    e=1;
end
if isnan(max(max(fA)))||isinf(max(max(fA))) %%No se puede calcular el exponencial de la matriz
    e=1;
end
