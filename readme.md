# NcclMatrixMultiplication
En este repositorio se pueden encontrar diversos proyectos.
## 1. NcclMatrixMultiplication
Este el es el proyecto principal donde se pueden encontrar la libreriía capaz de multiplicar matrices entre si mediante diversas **GPUs** con **CUDA** y **NCCL**. Además, de realizar otras operaciones como: Cambio de signo a toda la matri (−A), Operaciones de escalares con la matriz (3 ∗ A, A ∗ 3, A/3, A∗ = 3, A/ = 3, gemm personalizado), Operaciones con la matriz identidad (3 + A, A + 3, A+ = 3, 3 − A, A − 3, A− = 3), Suma y restas entre matrices (A + B, A+ = B, B − A, A− = B, axpy personalizado), Norma 1.

Si se desea compilar se hace mediante el siguiente comando `mkdir build && cd build && cmake ..`. Se puede indicar un directorio de insalación mediante `-DCMAKE_INSTALL_PREFIX=path`. Posteriormente se pueden compilar unos ejemplos que sirven de benchmark mediante `make versus1Gpu && make versus1Gpu`. [Aquí]([https://link](https://rodhuega.github.io/tfgMatrixNccl/doc/html/)) se puede consultar la documentación generada de la librería mediante DOxygen.

A continuación se muestra código de ejemplo de uso de la librería.
```C++
//Creación de objetos
NcclMultiplicationEnvironment<double> ncclMultEnv = NcclMultiplicationEnvironment<double>(gpuSizeWorldArgument, gpuRoot, opt, debugMatrix);
MatrixMain<double> ma = MatrixMain<double>(&ncclMultEnv, rowsA, columnsA, matrixA);
MatrixMain<double> mb = MatrixMain<double>(&ncclMultEnv, rowsA, columnsA);
mb.setMatrixHostToFullValue(1);
// mb.setMatrixHost(matrixA);
//Multiplicaciones
ma=ma*mb;
ma*=ma;
ma.setAlphaGemm(3);
ma=ma*mb;
ma.setAlphaGemm(1);
//Sumas y restas entre matrices
ma=ma+ma;
ma+=mb;
ma=ma-mb;
ma-=mb;
ma.axpy(2,mb);
//Sumas y restas con la identidad
ma=mb+1; 
ma=1+mb;
ma+=1;
ma=mb-1;
ma=1-mb;
ma-=1;
//Operaciones con escalares
ma=ma/3;
ma/=3;
ma=ma*6;
ma*=ma;
//norma 1
double resNorm1=ma.norm1();
//Cambio de signo
ma=-ma;
//Obtención de atributos
int rowsC=ma.getRowsReal();
int columnsC=ma.getColumnsReal();
double *distributedRes=ma.getHostMatrix();
//double *distributedRes=MatrixUtilitiesCuda<double>::matrixMemoryAllocationCPU(rowsC, columnsC);
//ma.getHostMatrixInThisPointer(distributedRes);
MatrixUtilitiesCuda<double>::printMatrix(rowsC, columnsC, distributedRes);
```

Si se desea usar mediante Matlab se debe de compilar de la siguiente forma: `mkdir build && cd build && cmake .. -DBUILD_FOR_MATLAB=ON -DMatlab_ROOT_DIR="/usr/local/MATLAB/R2018a"`. Importante señalar que el directorio de `-DMatlab_ROOT_DIR` puede que sea diferente. Para generar el fichero mex sera necesario usar `make call_gpu`. En caso de que de error de ejecución al usar el `.mex` generado puede que sea necesario que se lance Matlab con `LD_PRELOAD`. Por ejemplo `LD_PRELOAD="/usr/local/lib64/libstdc++.so.6 matlab` o `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 matlab`.

Código de ejemplo usando matlab:
```matlab
call_gpu('init',f,metodo_f,A);
a(1)=call_gpu('norm1',1);
pA{2}=call_gpu('power');
nProd=call_gpu('evaluate',p);
call_gpu('scale',s);
call_gpu('unscale',s);
call_gpu('free',1);
fA = call_gpu('finalize');
call_gpu('destroy');
```

## 2. MpiMatrixMultiplicationCpp
Este proyecto fue usado como experiencia puente antes de crear NcclMatrixMultiplication. No se va a detallar en profundidad ya que solo ejecuta un breve ejemplo de 2 multiplicacioens que se indican en el lanzamiento del programa. Realiza la multiplicación de matrices mediante SUMMA utilizando **Intel MKL** y **MPI**. Se compila mediante `make`. Puede que sea necesario editarlo para cambiar la path de Intel MKL.

## 3. ncclPruebas
Programa que ejecuta una serie de operaciones de CUDA y NCCL para medir cuanto tardan en ejecutarse dependiendo del tamaño de una matriz.

