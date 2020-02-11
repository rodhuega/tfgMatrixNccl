#include "MatrixMain.h"

MatrixMain::MatrixMain(int x,int y)
{
  gx = x;
  gy = y;
}

int MatrixMain::getSum()
{
  return gx + gy;
}