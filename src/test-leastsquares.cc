#include <dune/vem/misc/leastSquares.hh>

#include <dune/common/dynvector.hh>
#include <dune/common/dynmatrix.hh>

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


int main()
{
    typedef Dune::DynamicVector< double > Vector;

    std::vector< Vector > bVec, dVec, solnVec;
    Dune::DynamicMatrix< double > A, C;
    Dune::DynamicVector< double > b(3,0), d(3), x(3,0);
    Dune::DynamicVector< double > exactSoln(3,0);

    const int matrixDim = 3;

    A.resize( matrixDim, matrixDim, 0);
    C.resize( matrixDim, matrixDim, 1);

//
//    std::cout << "matrix A has columns: "<< A.cols() << std::endl;
//    std::cout << "matrix A has rows: "<< A.rows() << std::endl;
//
//    std::cout << "matrix C has columns: "<< C.cols() << std::endl;
//    std::cout << "matrix C has rows: "<< C.rows() << std::endl;

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

//    int count = 1;
//    for (unsigned int i = 0; i < matrixDim; ++i)
//        for ( unsigned int j=0; j < matrixDim; ++j) {
//            C[i][j] = count;
//            count++;
//        }

    std::cout << "Constraint C matrix: " << std::endl;
    printMatrix(C);


    b[2] = 1;
    exactSoln[0] = 0.75;
    exactSoln[1] = 1.5;
    exactSoln[2] = -0.25;
//    C.mv(b,d);

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

    bVec.push_back(b);
    dVec.push_back(d);
    solnVec.push_back(x);

//    std::cout << bVec.size() << std::endl;

    // define class member of least squares
    auto leastSquaresMinimizer = Dune::Vem::LeastSquares(A,C);

//    std::vector< Dune::DynamicVector< double >> bVec, dVec, solnVec;

//    std::vector< Dune::DynamicVector<double>> ;

//    bVec.push_back(b);
//    dVec.push_back(d);
//    solnVec.resize(bVec.size(),Dune::DynamicVector<double> (3,0));
//
//    for (unsigned int k=0; k<solnVec[0].size(); ++k)
//        std::cout << solnVec[0][k] << std::endl;

//    solnVec = leastSquaresMinimizer.leastSquaresSolver(b,d);

//    std::cout << "I reached here" << std::endl;
//    invoke least squares solver
    leastSquaresMinimizer.leastSquaresSystem(bVec, dVec, solnVec);

//    Dune::DynamicVector< double > llsSolnVec = solnVec[0];

    for ( unsigned int k = 0; k < solnVec[0].size(); ++k )
        std::cout << solnVec[0][k] << std::endl;

    // compare with exact solution
    const double tol = 1e-10;

    double error = 0;

    assert(solnVec[0].size() == exactSoln.size());

    // norm llsSolnVec - exactSoln;
    for (unsigned int i = 0; i < solnVec[0].size(); ++i )
    {
        error += (solnVec[0][i]-exactSoln[i])*(solnVec[0][i]-exactSoln[i]);
    }

    error = sqrt(error);
    std::cout << "error: " << error << std::endl;

    // define tolerance
    if (error < tol)
    {
        return 0;
    }

    return -1;

}