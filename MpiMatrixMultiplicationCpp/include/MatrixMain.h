#ifndef MatrixMain_H
#define MatrixMain_H

#include <iostream>
#include <fstream>
#include <random>
#include <mpi.h>
#include "MatrixUtilities.h"

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
  std::ifstream file;
  

public:
  MatrixMain(int rows, int columns, int boundLower, int boundUpper);
  MatrixMain(char *filename);
  int getRowsReal();
  int getRowsUsed();
  int getColumnsReal();
  int getColumnsUsed();
  double *getMatrix();
  void setRowsUsed(int rowsUsed);
  void setColumnsUsed(int columnsUsed);
  void fillMatrix(bool isRandom);
};

#endif