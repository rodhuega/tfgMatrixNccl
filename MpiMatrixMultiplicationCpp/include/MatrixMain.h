#ifndef MatrixMain_H
#define MatrixMain_H

#include <iostream>
#include <fstream>
#include <random>
#include <mpi.h>
#include "MatrixUtilities.h"

template <class Toperation>
class MatrixMain
{
private:
  int rowsReal;
  int rowsUsed;
  int columnsReal;
  int columnsUsed;
  int boundLower;
  int boundUpper;
  Toperation *matrix;
  std::ifstream file;
  

public:
  MatrixMain(int rows, int columns, int boundLower, int boundUpper);
  MatrixMain(const char *filename);
  int getRowsReal();
  int getRowsUsed();
  int getColumnsReal();
  int getColumnsUsed();
  Toperation *getMatrix();
  void setRowsUsed(int rowsUsed);
  void setColumnsUsed(int columnsUsed);
  void fillMatrix(bool isRandom);
  // void dummyFunction();

};
#endif