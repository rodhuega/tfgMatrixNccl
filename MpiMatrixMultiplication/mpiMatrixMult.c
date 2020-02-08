#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define blockNSize 2
#define N blockNSize *blockNSize

void MostrarMatrizDoblePuntero(int filas, int columnas, double **A)
{
    int i, j;
    for (i = 0; i < filas; ++i)
    {
        for (j = 0; j < columnas; ++j)
        {
            printf("%lf ", A[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void MostrarMatrizPuntero(int filas, int columnas, double *A)
{
    int i, j;
    for (i = 0; i < filas; ++i)
    {
        for (j = 0; j < columnas; ++j)
        {
            printf("%f ", A[i* filas + j]);
        }
        printf("\n");
    }
    printf("\n");
}



void RealizarMultiplicacion(int cpuRank, double *A, double *B, double *C)
{
    int i, j, k, sum;
    for (i = 0; i < blockNSize; i++)
    {
        sum = 0;
        for (j = 0; j < blockNSize; j++)
        {
            sum = 0;
            for (k = 0; k < blockNSize; k++)
            {
                sum += A[i * blockNSize + j] * B[i + blockNSize * j];
            }
            C[i*blockNSize+j]= sum;
        }
    }
}

void copiarMiniMatriz(int primeraFila,int primeraColumna,double **A,double* a_local)
{
    int i;
    for(i=0;i<blockNSize;i++)
    {
        memcpy(a_local+i*blockNSize, &A[primeraFila][primeraColumna], sizeof(double)*blockNSize);
        primeraFila+=1;
    }
}

void RealizarMultiplicacionBloque(int cpuRank, int blockNumberOfElements, double **A, double **B, double *c_local)
{
    int i, j;
    //Realizamos las dos multiplicaciones necesarias
    double a1_local[blockNSize][blockNSize];
    double a2_local[blockNSize][blockNSize];
    double b1_local[blockNSize][blockNSize];
    double b2_local[blockNSize][blockNSize];
    double c1_local[blockNSize][blockNSize];
    double c2_local[blockNSize][blockNSize];
    if (cpuRank == 0)
    {
        copiarMiniMatriz(0,0,A,a1_local);
        copiarMiniMatriz(0,2,A,a2_local);
        copiarMiniMatriz(0,0,B,b1_local);
        copiarMiniMatriz(2,0,B,b2_local);
        printf("HE COPIADO BIEN\n");
    }
    if (cpuRank == 1)
    {
        copiarMiniMatriz(0,0,A,a1_local);
        copiarMiniMatriz(0,2,A,a2_local);
        copiarMiniMatriz(0,2,B,b1_local);
        copiarMiniMatriz(2,2,B,b2_local);
    }
    if (cpuRank == 2)
    {
        copiarMiniMatriz(2,0,A,a1_local);
        copiarMiniMatriz(2,2,A,a2_local);
        copiarMiniMatriz(0,0,B,b1_local);
        copiarMiniMatriz(2,0,B,b2_local);
    }
    if (cpuRank == 3)
    {
        copiarMiniMatriz(2,0,A,a1_local);
        copiarMiniMatriz(2,2,A,a2_local);
        copiarMiniMatriz(0,2,B,b1_local);
        copiarMiniMatriz(2,2,B,b2_local);
    }
    RealizarMultiplicacion(cpuRank, a1_local, b1_local, c1_local);
    RealizarMultiplicacion(cpuRank, a2_local, b2_local, c2_local);
    //Sumamos los productos de esas multiplicaciones
    for (i = 0; i < blockNSize; i++)
    {
        for (j = 0; j < blockNSize; j++)
        {
            c_local[blockNSize * i + j] = c1_local[i][j] + c2_local[i][j];
        }
    }
}

double** LeerMatriz(char* nombreFichero)
{
    FILE *punteroFichero;
    punteroFichero =fopen(nombreFichero,"r");
    int i,j;
    int filas, columnas;
    fscanf(punteroFichero, "%d",&filas);
    fscanf(punteroFichero, "%d",&columnas);
    double** matriz=malloc(sizeof(double)*filas);
     
    for(i=0 ;i<filas; i++)
    {
        matriz[i]=malloc(sizeof(double)*columnas);
        for(j=0; j<columnas; j++)
        {
            fscanf(punteroFichero, "%lf",&matriz[i][j]);
            // printf("%lf ", matriz[i][j]);
        }
        printf("\n");
    }
    return matriz;
}

int main(int argc, char *argv[])
{
    int cpuRank, cpuSize, blockNumberOfElements;
    double c[N][N];
    double c_local[blockNSize][blockNSize];
    double** a= LeerMatriz("A.txt");
    double** b= LeerMatriz("B.txt");
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cpuSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);
    MPI_Datatype matrixLocalType;
    blockNumberOfElements = N * N / cpuSize;
    int send_counts[4] = {1, 1, 1, 1};
    int blocks[4] = {
        0, blockNSize,
        N*blockNSize, N*blockNSize + blockNSize
    };

    if (cpuRank == 0) {
        int sizes[2] = {N, N};
        int subsizes[2] = {blockNSize, blockNSize};
        int starts[2] = {0, 0};
        MPI_Type_create_subarray(2, sizes, subsizes, starts,
            MPI_ORDER_C, MPI_DOUBLE, &matrixLocalType);

        int double_size;
        MPI_Type_size(MPI_DOUBLE, &double_size);
        MPI_Type_create_resized(matrixLocalType, 0, 1*double_size, &matrixLocalType);
        MPI_Type_commit(&matrixLocalType);
    }

    RealizarMultiplicacionBloque(cpuRank,blockNumberOfElements,a,b,c_local);
 
    MPI_Gatherv(c_local, blockNumberOfElements, MPI_DOUBLE, c, send_counts, blocks, matrixLocalType, 0, MPI_COMM_WORLD);
    if(cpuRank==0)
    {
        MPI_Type_free(&matrixLocalType);
        MostrarMatrizPuntero(N,N,(double*)c);
    }
    MPI_Finalize();
}
