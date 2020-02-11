#include "MatrixBlock.h"

MatrixBlock::MatrixBlock(int x,int y)
{
  gx = x;
  gy = y;
}

int MatrixBlock::getSum()
{
  return gx + gy;
}