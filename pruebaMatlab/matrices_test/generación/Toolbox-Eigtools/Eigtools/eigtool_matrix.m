function f=eigtool_matrix()
    % Dense EigTool demos.
%   1 airy_demo            - An Airy operator.
%   2 basor_demo           - Toeplitz matrix (Basor-Morrison).
%   3 chebspec_demo        - First order Chebyshev differentiation matrix.
%   4 companion_demo       - A companion matrix.
%   5 convdiff_demo        - 1-D convection diffusion operator.
%   6 davies_demo          - Davies' example.
%   7 demmel_demo          - Demmel's matrix.
%   8 frank_demo           - The Frank matrix.
%   9 gallery3_demo        - The matrix `gallery(3)'.
%   10 gallery5_demo        - The matrix `gallery(5)'.
%   11 gaussseidel_demo     - Gauss-Seidel iteration matrices.
%   12 godunov_demo         - Godunov's matrix.
%   13 grcar_demo           - Grcar's matrix.
%   14 hatano_demo          - Hatano-Nelson example.
%   15 kahan_demo           - The Kahan matrix.
%   16 landau_demo          - An application from lasers.
%   17 orrsommerfeld_demo   - An Orr-Sommerfeld operator.
%   18 random_demo          - A matrix with random entries.
%   19 randomtri_demo       - An upper triangular matrix with random entries.
%   20 riffle_demo          - The riffle shuffle matrix.
%   21 transient_demo       - A matrix with transient behaviour.
%   22 twisted_demo         - A `twisted-Toeplitz' matrix.
% Sparse EigTool demos.
%   23 dwave_demo           - The dialectric waveguide matrix from Matrix Market.
%   24 convdiff_fd_demo     - A convection diffusion operator (finite-differences).
%   25 pde_demo             - An elliptic PDE from Matrix Market.
%   26 markov_demo          - A Markov chain transition matrix.
%   27 olmstead_demo        - The Olmstead model from Matrix Market.
%   28 rdbrusselator_demo   - The reaction-diffusion Brusellator model from Matrix Market.
%   29 sparserandom_demo    - A sparse random matrix.
%   30 skewlap3d_demo       - A non-symmetric 3D Laplacian.
%   31 supg_demo            - An SUPG matrix.
%   32 tolosa_demo          - The Tolosa matrix from Matrix Market.
i=0;
i=i+1;
f{i}='airy_demo';
i=i+1;
f{i}='basor_demo';
i=i+1;
f{i}='chebspec_demo';
i=i+1;
f{i}='companion_demo';
i=i+1;
f{i}='convdiff_demo';
i=i+1;
f{i}='davies_demo';
i=i+1;
f{i}='demmel_demo';
i=i+1;
f{i}='frank_demo';
i=i+1;
f{i}='gaussseidel_demo1';%Modificado del original. No converge para n>=4
i=i+1;
f{i}='godunov_demo';
i=i+1;
f{i}='grcar_demo';
i=i+1;
f{i}='hatano_demo';
i=i+1;
f{i}='kahan_demo';
i=i+1;
f{i}='landau_demo';
i=i+1;
f{i}='orrsommerfeld_demo';
i=i+1;
f{i}='random_demo';
i=i+1;
f{i}='randomtri_demo';
i=i+1;
f{i}='riffle_demo';
i=i+1;
f{i}='transient_demo';
i=i+1;
f{i}='twisted_demo';
%'dwave_demo';%solo sirve para n=2048
%'convdiff_fd_demo';se queda bloqueado para n=10
%'pde_demo';N can be 900 or 2961
%f{23}='markov_demo';Problemas con lo siguientes (comprobado hasta 23, pero
%       %me imagino que sigue igual
%f{24}='olmstead_demo';
%f{25}='rdbrusselator_demo';
%f{26}='sparserandom_demo';
%f{27}='skewlap3d_demo';
%f{28}='supg_demo';
%f{29}='tolosa_demo';
end