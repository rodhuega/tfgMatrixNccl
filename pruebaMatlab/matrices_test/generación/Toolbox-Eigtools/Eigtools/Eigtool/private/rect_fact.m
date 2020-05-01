function [S,T] = rect_fact(A)

% function [S,T] = rect_fact(A)
%
% Compute the long-triangular form of the matrix. This should be called
% before psa_computation.m which requires a long-triangular matrix as input.
%
% A           the matrix to factor
% 
% S           the triangular factors
% T           the triangular factor (if m<2n)

% Version 2.1 (Mon Mar 16 00:24:50 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  [m,n] = size(A);

  if m>=2*n,  % Use QR form
    [Q,R] = qr(A(n+1:end,:),0);
    S = [A(1:n,:); R];
    T = 1;
  else        % Use QZ form
    T = eye(m,n);
    [S,T,Q,Z] = qz(A(end-n+1:end,:),T(end-n+1:end,:));
    S = [A(1:m-n,:)*Z; S];
    T = [Z(1:m-n,:); T];
  end;
