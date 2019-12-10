#include <dune/vem/misc/leastSquares.hh>

#include <dune/common/dynvector.hh>
#include <dune/common/dynmatrix.hh>


template <class Matrix>
Matrix emptyMatrix()
{
    // return matrix with no size
    Matrix A;
//    std::cout << "size of empty matrix " << A.size() << std::endl;
    return A;
}

template< class Matrix >
void printMatrix(const Matrix &A)
{
    for(unsigned int i = 0; i < A.rows(); ++i) {
        for (unsigned int j = 0; j < A.cols(); ++j) {
            std::cout << A[i][j];
        }
        std::cout << std::endl;
    }

}

template< class Vector>
void printVector(const Vector &x)
{
    for(unsigned int i = 0; i < x.size(); ++i)
        std::cout << x[i] << std::endl;
}


int main( int argc, char **argv )
{
    typedef Dune::DynamicVector< double > Vector;

    // test 0 or 1
    int testNumber;

    if ( argc < 2 ){
//        testNumber = 0;
        std::cerr << "please choose test" << std::endl;
        return -1;
    }

    testNumber = atoi(argv[1]);
    const double tol = 1e-10;

    Dune::DynamicMatrix< Vector > R;
    Vector v1(2),v2(2),v3(2),v4(2);
    v1[0] = 0;
    v1[1] = 1;
    v2[0] = 1;
    v2[1] = 0;
    v3[0] = 1;
    v3[1] = 1;
    v4[0] = 2;
    v4[1] = 1;

    R.resize(2,2);
    R[0][0] = v1;
    R[0][1] = v2;
    R[1][0] = v3;
    R[1][1] = v4;

    auto colR = Dune::Vem::columnVector( R, 1 );
    Vector d = colR.concatenateColumn<Vector>();

    printVector(d);

    // 0
    // 1
    // 1
    // 1


//  Dune::Vem::ColumnVector<Dune::DynamicMatrix<std::vector<double>>> columnVector(R,1);

//    d = expandRcol.expand();

//    printVector(d);




    if (testNumber == 0){
        std::vector< Vector > bVec, dVec, solnVec;
        Dune::DynamicMatrix< double > A, C;
        Dune::DynamicVector< double > b(3,0), d(2), x(3,0);
        Dune::DynamicMatrix< double > valueProj;
        Dune::DynamicVector< double > exactSoln(3,0);



        const int matrixDim = 3;

        valueProj.resize( 3, 2, 0);
        A.resize( matrixDim, matrixDim, 0);
        C.resize( 3, 3, 1);

        // set A equal to identity
        for (unsigned int i = 0; i < matrixDim; ++i)
            A[i][i] = 1;

        std::cout << "Least squares A matrix: " << std::endl;
        printMatrix(A);

        // initialise the matrix C
        C[0][1] = 2;
        C[0][2] = 3;
        C[1][1] = 3;

        std::cout << "Constraint C matrix: " << std::endl;
        printMatrix(C);

        Dune::Vem::BlockMatrix<Dune::DynamicMatrix<double>> blockMatrix(C,2);

        blockMatrix[3];

        printVector(blockMatrix[3]);



//    Dune::Vem::ColumnVector<Dune::DynamicMatrix<double>> columnVector(C,1);
//    columnVector[0] = 5;
//    columnVector[1] = 6;
//    columnVector[2] = 7;

//        std::cout << "column vec C" << std::endl;
//        printMatrix(C);

        b[2] = 1;
        exactSoln[0] = 0.0741;
        exactSoln[1] = 0.4074;
        exactSoln[2] = 0.7037;

        d[0] = 3;
        d[1] = 2;

        std::cout << "Vector b: " << std::endl;
        printVector(b);

        std::cout << "Vector d: " << std::endl;
        printVector(d);

        std::cout << "Vector x: " << std::endl;
        printVector(x);

        std::cout << "Vector exact solution: " << std::endl;
        printVector(exactSoln);

        auto leastSquaresMinimizer = Dune::Vem::LeastSquares(A,C);
        Dune::DynamicVector< double > solutionVector;


        for( unsigned int i = 0; i < valueProj.cols(); ++i){
            auto columnValueProj = Dune::Vem::ColumnVector(valueProj,i);

            columnValueProj = leastSquaresMinimizer.solve(b,d);
//            printVector(solutionVector);
//            columnValueProj = (solutionVector);
        }

        std::cout << "value projection " << std::endl;
        printMatrix(valueProj);

        bVec.push_back(b);
        dVec.push_back(d);
        solnVec.push_back(x);

        // define class member of least squares

        leastSquaresMinimizer.solve(bVec, dVec, solnVec);

        for ( unsigned int k = 0; k < solnVec[0].size(); ++k )
            std::cout << solnVec[0][k] << std::endl;

        // compare with exact solution
        double error = 0;

        assert(solnVec[0].size() == exactSoln.size());

        // norm llsSolnVec - exactSoln;
        for (unsigned int i = 0; i < solnVec[0].size(); ++i )
            error += (solnVec[0][i]-exactSoln[i])*(solnVec[0][i]-exactSoln[i]);

        error = sqrt(error);
        std::cout << "error: " << error << std::endl;

        if (error < tol) {
            return 0;
        }

        return -1;

    }
    if (testNumber == 1){
        std::vector< Vector > bVec, dVec, solnVec;
        Dune::DynamicMatrix< double > A, C;
        Dune::DynamicVector< double > b(4,0), d(2), x(3,0);
        Dune::DynamicMatrix< double > valueProj;
        Dune::DynamicVector< double > exactSoln(3,0);


        std::cout << "Sie of C " << C.size() << std::endl;

        const int matrixDim = 3;

        valueProj.resize( 3, 2, 0);
        A.resize( 4, matrixDim, 0);
//        C.resize( 2, matrixDim, 1);

        // set A equal to identity
        for (unsigned int i = 0; i < matrixDim; ++i)
            A[i][i] = 1;

        A[3][0] = 1;
        A[3][1] = 1;

        std::cout << "Least squares A matrix: " << std::endl;
        printMatrix(A);

        // initialise the matrix C
//        C[0][1] = 2;
//        C[0][2] = 3;
//        C[1][1] = 3;

        std::cout << "Constraint C matrix: " << std::endl;
        printMatrix(C);

//    Dune::Vem::ColumnVector<Dune::DynamicMatrix<double>> columnVector(C,1);
//    columnVector[0] = 5;
//    columnVector[1] = 6;
//    columnVector[2] = 7;

//        std::cout << "column vec C" << std::endl;
//        printMatrix(C);

        b[3] = 1;
        exactSoln[0] = 0.392405063291139;
        exactSoln[1] = 0.316455696202532;
        exactSoln[2] = 0.658227848101266;

        d[0] = 3;
        d[1] = 2;

        std::cout << "Vector b: " << std::endl;
        printVector(b);

        std::cout << "Vector d: " << std::endl;
        printVector(d);

        std::cout << "Vector x: " << std::endl;
        printVector(x);

        std::cout << "Vector exact solution: " << std::endl;
        printVector(exactSoln);

        auto leastSquaresMinimizer = Dune::Vem::LeastSquares(A);

        std::cout << "I reached here" << std::endl;

        for( unsigned int i = 0; i < valueProj.cols(); ++i){
            auto colVecAdjustment = Dune::Vem::ColumnVector(valueProj,i);

//        Dune::Vem::ColumnVector<Dune::DynamicMatrix<double>> columnVectorVP(valueProj,i);
            leastSquaresMinimizer.solve(b,d);
        }

        std::cout << "value projection " << std::endl;
        printMatrix(valueProj);

        bVec.push_back(b);
        dVec.push_back(d);
        solnVec.push_back(x);

        // define class member of least squares

        leastSquaresMinimizer.solve(bVec, dVec, solnVec);

        for ( unsigned int k = 0; k < solnVec[0].size(); ++k )
            std::cout << solnVec[0][k] << std::endl;

        // compare with exact solution
        double error = 0;

        assert(solnVec[0].size() == exactSoln.size());

        // norm llsSolnVec - exactSoln;
        for (unsigned int i = 0; i < solnVec[0].size(); ++i )
            error += (solnVec[0][i]-exactSoln[i])*(solnVec[0][i]-exactSoln[i]);

        error = sqrt(error);
        std::cout << "error: " << error << std::endl;

        if (error < tol) {
            return 0;
        }

        return -1;

    }
  if (testNumber == 2){
    std::vector< Vector > bVec, dVec, solnVec;
    Dune::DynamicMatrix< double > A, C(2,3,1);
    Dune::DynamicVector< double > b(3), d(2), x(3,0);
    Dune::DynamicMatrix< double > valueProj;
//    Dune::DynamicVector< double > exactSoln(3,0);

    // initialise the matrix C
        C[0][1] = 2;
        C[0][2] = 3;
        C[1][1] = 3;

    std::cout << "Constraint C matrix: " << std::endl;
    printMatrix(C);

    d[0] = 3;
    d[1] = 2;

    std::cout << "Vector d: " << std::endl;
    printVector(d);

    auto leastSquaresMinimizer = Dune::Vem::LeastSquares(A,C);


    x = leastSquaresMinimizer.solve(b,d);

    printVector(x);

//    if (error < tol) {
      return 0;
//    }

//    return -1;

  }
    else {

        std::cout << "test number 0 not chosen" << std::endl;
        std::vector< Vector > bVec, dVec, solnVec;
        Dune::DynamicMatrix< double > A, C;
        Dune::DynamicVector< double > b(3,0), d(3), x(3,0);
        Dune::DynamicVector< double > exactSoln(3,0);

        Dune::DynamicMatrix< double > valueProj;

        valueProj.resize( 3, 2, 0);


        const double tol = 1e-10;

        const int matrixDim = 3;

        A.resize( matrixDim, matrixDim, 0);
        C.resize( matrixDim, matrixDim, 1);

        // set A equal to identity
        for (unsigned int i = 0; i < matrixDim; ++i)
            A[i][i] = 1;

        std::cout << "Least squares A matrix: " << std::endl;
        printMatrix(A);

        // initialise the matrix C
        C[0][1] = 2;
        C[0][2] = 3;
        C[1][1] = 3;
        C[2][0] = 2;
        C[2][2] = 4;

        std::cout << "Constraint C matrix: " << std::endl;
        printMatrix(C);


//    Dune::Vem::ColumnVector<Dune::DynamicMatrix<double>> columnVector(C,1);
//    columnVector[0] = 5;
//    columnVector[1] = 6;
//    columnVector[2] = 7;

        std::cout << "column vec C" << std::endl;
        printMatrix(C);



        b[2] = 1;
        exactSoln[0] = 0.75;
        exactSoln[1] = 1.5;
        exactSoln[2] = -0.25;

        d[0] = 3;
        d[1] = 5;
        d[2] = 2;

        std::cout << "Vector b: " << std::endl;
        printVector(b);

        std::cout << "Vector d: " << std::endl;
        printVector(d);

        std::cout << "Vector x: " << std::endl;
        printVector(x);

        std::cout << "Vector exact solution: " << std::endl;
        printVector(exactSoln);

        auto leastSquaresMinimizer = Dune::Vem::LeastSquares(A,C);


        for( unsigned int i = 0; i < valueProj.cols(); ++i){
            auto colVecAdjustment = Dune::Vem::ColumnVector(valueProj,i);

//        Dune::Vem::ColumnVector<Dune::DynamicMatrix<double>> columnVectorVP(valueProj,i);

            leastSquaresMinimizer.solve(b,d);
        }

        std::cout << "value projection " << std::endl;
        printMatrix(valueProj);

        bVec.push_back(b);
        dVec.push_back(d);
        solnVec.push_back(x);

        // define class member of least squares

        leastSquaresMinimizer.solve(bVec, dVec, solnVec);

        for ( unsigned int k = 0; k < solnVec[0].size(); ++k )
            std::cout << solnVec[0][k] << std::endl;

        // compare with exact solution
        double error = 0;

        assert(solnVec[0].size() == exactSoln.size());

        // norm llsSolnVec - exactSoln;
        for (unsigned int i = 0; i < solnVec[0].size(); ++i )
            error += (solnVec[0][i]-exactSoln[i])*(solnVec[0][i]-exactSoln[i]);

        error = sqrt(error);
        std::cout << "error: " << error << std::endl;

        if (error < tol) {
            return 0;
        }

        return -1;
    }
}