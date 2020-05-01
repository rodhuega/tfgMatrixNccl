function F = frank_demo(N)

%   F = FRANK_DEMO(N) returns matrix F of dimension N.
%
%   The Frank matrix is also in the MATLAB gallery, and can be
%   created by the command F = gallery('frank',N)
%
%   F is the Frank matrix, a classic example of a matrix whose
%   eigenvalues are ill-conditioned and hence difficult to
%   compute [1,2]. What it illustrates is that one cannot 
%   expect to compute eigenvalues better than a set of
%   eps-pseudoeigenvalues for some eps on the order of 
%   machine precision. This is seen at the left of the
%   spectrum, where some eigenvalues are complex, even
%   though all the eigenvalues can be proved to be real. The
%   pseudospectra are shown in [3].
%
%   [1]: G. H. Golub and J. H. Wilkinson, "Ill-conditioned
%        eigensystems and the computation of the Jordan
%        canonical form", SIAM Review 18, pp. 578-619, 1976.
%   [2]: J. H. Wilkinson, "The Algebraic Eigenvalue Problem",
%        Clarendon Press, Oxford, 1965.
%   [3]: L. N. Trefethen, "Psuedospectra of matrices", in
%        "Numerical Analysis 1991" (Dundee 1991), Longman Sci.
%        Tech., Harlow, 1992, 234-266.

% Version 2.1 (Mon Mar 16 00:24:49 CDT 2009)
% Copyright 2002 - 2009 by Tom Wright; maintained by Mark Embree (embree@rice.edu)

  F = triu(repmat(N:-1:1,N,1))+diag(N-1:-1:1,-1);
