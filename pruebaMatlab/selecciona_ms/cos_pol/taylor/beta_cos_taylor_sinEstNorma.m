function Beta_min = beta_cos_taylor_sinEstNorma(b,d,m,q)
% Beta_min = beta_cos_taylor_sinEstNorma(b,d,m,q) 
% Computes beta_min with only products of norms of known matrix powers
% m = order of approximation (2,4,6,9,12,16,>=20)
% d = norm(B^k,1), k=1:q
% b = norm(B^k,1)^1/k, k=1:q
switch m
    case 2                                             %m = 2
        Beta_min = (d(2)*d(1))^(1/3);
    case 4                                             %m = 4
        Beta_min = (d(2)^2*d(1))^(1/5);
    case 6                                             %m = 6
        if q == 2
            Beta_min = min(d(2)^3*d(1))^(1/7);
        elseif q == 3
            if b(2) <= b(3)
                Beta_min = min(d(2)^2*d(3),d(1)*d(6))^(1/7);
            else
                Beta_min = max(min(d(2)^2*d(3),d(1)*d(6))^(1/7),(d(6)*d(2))^(1/8));
            end
        end    
    case 9                                             %m = 9
        if b(2) <= b(3)
            Beta_min = (d(2)^3*d(3))^(1/9);
        else
            Beta_min = max(min(d(2)^2*d(3)^2,d(3)^3*d(1))^(1/10),(d(3)^3*d(2))^(1/11));
        end
    case 12                                              %m = 12       
         if q == 3 
            if b(2) <= b(3)
                Beta_min = (d(2)^5*d(3))^(1/13);
            else
                Beta_min = min(d(3)^4*d(1),d(3)^3*d(2)^2)^(1/13);
            end
        elseif q == 4
            if b(3) <= b(4)
                Beta_min = max((d(3)^3*d(4))^(1/13),min(d(3)^2*d(4)^2,d(3)^4*d(2))^(1/14));
            else
                Beta_min = max((d(4)^2*min(d(3)*d(2),d(4)*d(1)))^(1/13),(d(4)^2*d(6))^(1/14));
            end                    
        end       
    case 16                                              %m = 16
        if b(3) <= b(4)
            Beta_min = max((d(3)^4*d(4))^(1/16),min(d(3)^5*d(2),d(3)^3*d(4)^2)^(1/17));
        else
            Beta_min = max((d(4)^3*min(d(4)*d(1),d(3)*d(2)))^(1/17),(d(4)^3*d(6))^(1/18));
        end
    otherwise                                              %m >= 20
        if b(3) <= b(4)
            Beta_min = max(min(d(3)^6*d(2),d(3)^4*d(4)^2)^(1/20),(d(3)^6*d(4))^(1/22));
        else
            Beta_min = max(min(d(4)^5*d(1),d(4)^4*d(3)*d(2))^(1/21),(d(4)^4*d(6))^(1/22));
        end
end
if isnan(Beta_min)||isinf(Beta_min)
    error('beta_cos_taylor_sinEstNorma:NanInfEstimation',['Nan or inf appeared in bounding the norms of high matrix powers with products of the norms of known matrix powers, check matrix'])
end
end
