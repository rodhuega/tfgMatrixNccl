function Beta_min = beta_cos_taylor_sastre_sinEstNorma(b,d,m)
% Beta_min = beta_NoNormEst(b,d,m) 
% Computes beta_min with only products of norms of known matrix powers
% m = order of approximation (2,4,8,12,15)
% d = norm(B^k,1), k=1:q
% b = norm(B^k,1)^1/k, k=1:q
switch m
    case 2                                             %m = 2  (m+1,m+2)
        Beta_min = (d(2)*d(1))^(1/3);
    case 4                                             %m = 4  (m+1,m+2)
        Beta_min = (d(2)^2*d(1))^(1/5);
    case 8                                             %m = 8  (m,m+1)     
        Beta_min = min(d(2)^4*d(1))^(1/9);
    case 12                                            %m = 12
        if b(2) <= b(3)                                %(m,m+1)
            Beta_min = (d(2)^5*d(3))^(1/13);
        else                                           %(m,m+1,m+2)
            Beta_min = max(min(d(3)^4*d(1),d(3)^3*d(2)^2)^(1/13),(d(3)^4*d(2))^(1/14));
        end     
    case 15                                              %m = 15
        if b(2) <= b(3)                                  %(m,m+1)
            Beta_min = (d(2)^6*d(3))^(1/15);
        else                                             %(m,m+1,m+2)
            Beta_min = max((d(3)^4*min(d(3)*d(1),d(2)^2))^(1/16),(d(3)^5*d(2))^(1/17));
        end
end
if isnan(Beta_min)||isinf(Beta_min)
    error('beta_cos_taylor_sastre_sinEstNorma:NanInfEstimation',['Nan or inf appeared in bounding the norms of high matrix powers with products of the norms of known matrix powers, check matrix'])
end
end   