#ifndef MatrixBlock_H
#define MatrixBlock_H

class MatrixBlock
{
private:
    int deviceIdOwner;
    int rowFirstPosition;
    int columnFirstPosition;
    int rowLastPosition;
    int columnLastPosition;
    int rowsReal;
    int columnsReal;
    int rowsUsed;
    int columnsUsed;
    bool extendedRow;
    bool extendedColumn;
    double** matrix;

public:
    MatrixBlock(double** smallMatrix);
};

#endif