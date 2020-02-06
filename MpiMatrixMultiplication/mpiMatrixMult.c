#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define N 12

void print_results(char *prompt, double a[N][N]);

int main(int argc, char *argv[])
{
    int cpuId, cpiN,i,j, sum = 0;
    double a[N][N]={{181,   2, 121, 288, 211, 268, 147,  93,  38,  62, 194,  53},
       {158, 232, 131, 280, 216, 171,  40,  75, 199, 277,   7,  87},
       {230, 242, 124,  17,  10, 187, 252, 221,  69,  79,  28, 291},
       {238,  89,   0, 150, 287, 177, 121, 128, 275,  51,  31, 131},
       {122, 276,  19, 179,  12,   5, 242, 230,  12, 108, 129, 147},
       { 98,  90, 174,  34, 108, 280, 163, 229,   3,  45,  24, 148},
       {103, 114,  85, 210,   6,  75, 237, 148,  13,  59, 223, 138},
       {  6,   8, 214, 124, 295,  68,  88, 212,   6, 252,  11,  44},
       {233, 142,  25, 282, 257, 155,  33, 165, 236, 113,  88, 164},
       {207,  42,  13, 123,  41,  87, 175, 217, 197, 240, 167,  93},
       { 88, 259, 242, 281,   6,  85, 200,  44, 224, 265,  18, 167},
       {257, 263,   1, 112, 268,  32, 149,  30, 221, 252,  95, 118}};
    double b[N][N]={{181,   2, 121, 288, 211, 268, 147,  93,  38,  62, 194,  53},
       {158, 232, 131, 280, 216, 171,  40,  75, 199, 277,   7,  87},
       {230, 242, 124,  17,  10, 187, 252, 221,  69,  79,  28, 291},
       {238,  89,   0, 150, 287, 177, 121, 128, 275,  51,  31, 131},
       {122, 276,  19, 179,  12,   5, 242, 230,  12, 108, 129, 147},
       { 98,  90, 174,  34, 108, 280, 163, 229,   3,  45,  24, 148},
       {103, 114,  85, 210,   6,  75, 237, 148,  13,  59, 223, 138},
       {  6,   8, 214, 124, 295,  68,  88, 212,   6, 252,  11,  44},
       {233, 142,  25, 282, 257, 155,  33, 165, 236, 113,  88, 164},
       {207,  42,  13, 123,  41,  87, 175, 217, 197, 240, 167,  93},
       { 88, 259, 242, 281,   6,  85, 200,  44, 224, 265,  18, 167},
       {257, 263,   1, 112, 268,  32, 149,  30, 221, 252,  95, 118}};
    double c[N][N];
    double aa[N],cc[N],bb[N];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cpiN);
    MPI_Comm_rank(MPI_COMM_WORLD, &cpuId);

    //Idea que puede funcionar para nccl, broadcast de a y b, ya que no se puede hacer scatter
    //En a y en b despues en cada nodo coger los bloques que interese.
    //Otra idea pasar a y b con scatter por mpi y luego copiarlo al device

    MPI_Scatter(a, N*N/cpiN, MPI_DOUBLE, aa, N*N/cpiN, MPI_DOUBLE,0,MPI_COMM_WORLD);
    for(i = 0;i<N;i++)
    {
        MPI_Scatter(b, N*N/cpiN, MPI_DOUBLE, bb, N*N/cpiN, MPI_DOUBLE,0,MPI_COMM_WORLD);
    }

    //perform vector multiplication by all processes
    for (i = 0; i < N; i++)
      {
              for (j = 0; j < N; j++)
              {
                      sum = sum + aa[j] * b[j][i];  //MISTAKE_WAS_HERE               
              }
              cc[i] = sum;
              sum = 0;
      }

    MPI_Gather(cc, N*N/cpiN, MPI_DOUBLE, c, N*N/cpiN, MPI_DOUBLE, 0, MPI_COMM_WORLD);     
    MPI_Finalize();
    if (cpuId == 0)                         //I_ADDED_THIS
        print_results("C = ", c);
}

void print_results(char *prompt, double a[N][N])
{
    int i,j;
    for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                    printf(" %f", a[i][j]);
            }
            printf ("\n");
    }
    printf ("\n\n");
}