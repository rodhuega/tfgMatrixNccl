#ifndef MatrixMain_H
#define MatrixMain_H

#include <iostream>
#include <fstream>
#include <mpi.h>
using namespace std;
class MatrixMain
{
private:
  int rowsReal;
  int rowsUsed;
  int columnsReal;
  int columnsUsed;
  int boundLower;
  int boundUpper;
  double *matrix;
  ifstream file;

  void fillMatrix(bool isRandom);

public:
  MatrixMain(int rows, int columns, int boundLower, int boundUpper);
  MatrixMain(char *filename);
  int getRowsReal();
  int getRowsUsed();
  int getColumnsReal();
  int getColumnsUsed();
  double *getMatrix();
};

#endif