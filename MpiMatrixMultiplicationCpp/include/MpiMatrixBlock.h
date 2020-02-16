#ifndef MpiMatrixBlock_H
#define MpiMatrixBlock_H

class MpiMatrixBlock
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
    MpiMatrixBlock(double** smallMatrix);
};

#endif