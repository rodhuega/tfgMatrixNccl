// A2DD.cpp
#include "MatrixBlock.h"

A2DD::A2DD(int x,int y)
{
  gx = x;
  gy = y;
}

int A2DD::getSum()
{
  return gx + gy;
}