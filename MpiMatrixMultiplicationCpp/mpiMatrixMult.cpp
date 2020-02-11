#include <string>
#include <mpi.h>
#include <iostream>
#include <fstream>

using namespace std;

int blockNSize;

void PrintDoublePointerMatrix(int rows, int columns, double **A)
{
    int i, j;
    for (i = 0; i < rows; ++i)
    {
        for (j = 0; j < columns; ++j)
        {
            printf("%lf ", A[i][j]);
            cout << A[i][j] << " ";
        }
        cout<< endl;
    }
}

// void MostrarMatrizPuntero(int filas, int columnas, double *A)
// {
//     int i, j;
//     for (i = 0; i < filas; ++i)
//     {
//         for (j = 0; j < columnas; ++j)
//         {
//             printf("%f ", A[i* filas + j]);
//         }
//         printf("\n");
//     }
//     printf("\n");
// }



// void RealizarMultiplicacion(int cpuRank, double *A, double *B, double *C)
// {
//     int i, j, k, sum;
//     for (i = 0; i < blockNSize; i++)
//     {
//         sum = 0;
//         for (j = 0; j < blockNSize; j++)
//         {
//             sum = 0;
//             for (k = 0; k < blockNSize; k++)
//             {
//                 sum += A[i * blockNSize + j] * B[i + blockNSize * j];
//             }
//             C[i*blockNSize+j]= sum;
//         }
//     }
// }

// void copiarMiniMatriz(int primeraFila,int primeraColumna,double **A,double* a_local)
// {
//     int i;
//     for(i=0;i<blockNSize;i++)
//     {
//         memcpy(a_local+i*blockNSize, &A[primeraFila][primeraColumna], sizeof(double)*blockNSize);
//         primeraFila+=1;
//     }
// }

// void RealizarMultiplicacionBloque(int cpuRank, int blockNumberOfElements, double **A, double **B, double *c_local)
// {
//     int i, j;
//     //Realizamos las dos multiplicaciones necesarias
//     double a1_local[blockNSize][blockNSize];
//     double a2_local[blockNSize][blockNSize];
//     double b1_local[blockNSize][blockNSize];
//     double b2_local[blockNSize][blockNSize];
//     double c1_local[blockNSize][blockNSize];
//     double c2_local[blockNSize][blockNSize];
//     if (cpuRank == 0)
//     {
//         copiarMiniMatriz(0,0,A,a1_local);
//         copiarMiniMatriz(0,blockNSize,A,a2_local);
//         copiarMiniMatriz(0,0,B,b1_local);
//         copiarMiniMatriz(blockNSize,0,B,b2_local);
//     }
//     if (cpuRank == 1)
//     {
//         copiarMiniMatriz(0,0,A,a1_local);
//         copiarMiniMatriz(0,blockNSize,A,a2_local);
//         copiarMiniMatriz(0,blockNSize,B,b1_local);
//         copiarMiniMatriz(blockNSize,blockNSize,B,b2_local);
//     }
//     if (cpuRank == 2)
//     {
//         copiarMiniMatriz(blockNSize,0,A,a1_local);
//         copiarMiniMatriz(blockNSize,blockNSize,A,a2_local);
//         copiarMiniMatriz(0,0,B,b1_local);
//         copiarMiniMatriz(blockNSize,0,B,b2_local);
//     }
//     if (cpuRank == 3)
//     {
//         copiarMiniMatriz(blockNSize,0,A,a1_local);
//         copiarMiniMatriz(blockNSize,blockNSize,A,a2_local);
//         copiarMiniMatriz(0,blockNSize,B,b1_local);
//         copiarMiniMatriz(blockNSize,blockNSize,B,b2_local);
//     }
//     RealizarMultiplicacion(cpuRank, a1_local, b1_local, c1_local);
//     RealizarMultiplicacion(cpuRank, a2_local, b2_local, c2_local);
//     //Sumamos los productos de esas multiplicaciones
//     for (i = 0; i < blockNSize; i++)
//     {
//         for (j = 0; j < blockNSize; j++)
//         {
//             c_local[blockNSize * i + j] = c1_local[i][j] + c2_local[i][j];
//         }
//     }
    
// }

double** ReadMatrix(char* fileName,int* rowsM, int* columnsM)
{
    string line;
    ifstream file;
	file.open(fileName);
    int i,j;
    bool extendedRow,extendedColumn;
    int filasReal, columnsReal,rowsUsed,columnsUsed;
    file >> filasReal>>columnsReal;
    rowsUsed= filasReal%2?filasReal+1:filasReal;
    columnsUsed= columnsReal%2?columnsReal+1:columnsReal;
    cout<< rowsUsed<<endl;
    double** matriz=new double*[rowsUsed];
    for(i=0 ;i<rowsUsed; i++)
    {
        matriz[i]=new double[rowsUsed*columnsUsed];
        for(j=0; j<columnsUsed; j++)
        {
            file >> matriz[i][j];
            cout << matriz[i][j]<< " ";
        }
        cout << endl;
    }
    return matriz;
}

int main(int argc, char *argv[])
{
    int cpuRank, cpuSize;
    int rowsA,columnsA,rowsB,columnsB;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cpuSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);
    double** a;
    double** b;
    if (cpuRank==0)
    {
        a= ReadMatrix(argv[1],&rowsA,&columnsA);
        PrintDoublePointerMatrix(rowsA,columnsA,a);
    }
    MPI_Finalize();
}
