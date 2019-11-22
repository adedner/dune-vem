#ifndef DUNE_VEM_MISC_LEASTSQUARES_HH
#define DUNE_VEM_MISC_LEASTSQUARES_HH

#include <vector>

#include <dune/common/dynmatrix.hh>

#include <assert.h>

//using namespace std;

namespace Dune {

    namespace Vem {

        template< class Matrix >
        class LeastSquares {
            public:
                typedef typename Matrix::size_type Size;

                LeastSquares(const Matrix &llsMatrix, const Matrix &constraintMatrix)
                : llsMatrix_(llsMatrix), constraintMatrix_(constraintMatrix),
                  systemMatrixInv_(matrixSetUp())
                {
                }
                LeastSquares(const LeastSquares &source) = delete;

                template <class Vector>
                Vector leastSquaresSolver(const Vector &b, const Vector &d ){

                    assert( (b.size() == d.size()) );

                    Vector solution(b.size());
                    Vector systemVector = vectorSetUp(b,d);

                    Size systemMatrixDim = systemMatrixInv_.rows();

                    // since systemMatrix sqaure, rows = cols
                    Vector systemMultiply(systemMatrixDim, 0);


                    for ( Size i = 0; i < systemMatrixDim; ++i) {
                        for (Size j = 0; j < systemMatrixDim; ++j)
                            systemMultiply[i] += systemMatrixInv_[i][j] * systemVector[j];
                    }

                    for ( Size i = 0; i < solution.size(); ++i)
                        solution[i] = systemMultiply[i];

                    return solution;
                }

                template <class Vector>
                void leastSquaresSystem(const std::vector< Vector > &bVec,
                                        const std::vector< Vector > &dVec,
                                        std::vector< Vector > &solnVec){
                    // check dimensions match
                    assert( (bVec.size() == dVec.size()) && (dVec.size() == solnVec.size()));

                    for ( unsigned int i = 0; i < bVec.size(); ++i ){
                        solnVec[i] = leastSquaresSolver(bVec[i],dVec[i]);
                    }
                }

//                template< class Matrix >
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


            private:
                const Matrix &llsMatrix_;
                const Matrix &constraintMatrix_;
                const Matrix systemMatrixInv_;
//                const Size LLSMatrixNumRows_ = llsMatrix_.rows();
//                const Size LLSMatrixNumCols_ = llsMatrix_.cols();
//                const Size constraintMatrixNumCols_ = constraintMatrix_.cols();
//                const Size constraintMatrixNumRows_ = constraintMatrix_.rows();

                Size llsMatrixNumRows()
                {
                    return llsMatrix_.rows();
                }

                Size llsMatrixNumCols()
                {
                    return llsMatrix_.cols();
                }

                Size constraintMatrixNumRows()
                {
                    return constraintMatrix_.rows();
                }

                Size constraintMatrixNumCols()
                {
                    return constraintMatrix_.cols();
                }

                Matrix matrixSetUp()
                {
                    // construct the matrix [2A^T*A C^T ; C 0] needed for least squares solution
                    Matrix systemMatrix;

                    const Size LLSMatrixNumRows_ = llsMatrixNumRows();
                    const Size LLSMatrixNumCols_ = llsMatrixNumCols();
                    const Size constraintMatrixNumCols_ = constraintMatrixNumCols();
                    const Size constraintMatrixNumRows_ = constraintMatrixNumRows();

//                    std::cout << "matrix llsMatrix has columns: "<< LLSMatrixNumCols_ << std::endl;
//                    std::cout << "matrix llsMatrix has rows: "<< LLSMatrixNumRows_ << std::endl;
//
//                    std::cout << "matrix constraint has columns: "<< constraintMatrixNumCols_ << std::endl;
//                    std::cout << "matrix constraint has rows: "<< constraintMatrixNumRows_ << std::endl;

//                    std::cout << "matrix llsMatrix has columns: "<< llsMatrixNumCols() << std::endl;
//                    std::cout << "matrix llsMatrix has rows: "<< llsMatrixNumRows() << std::endl;
//
//                    std::cout << "matrix constraint has columns: "<< constraintMatrixNumCols() << std::endl;
//                    std::cout << "matrix constraint has rows: "<< constraintMatrixNumRows() << std::endl;

                    // check dimensions compatible

                    assert( llsMatrixNumCols() == constraintMatrixNumCols() );
                    assert( ( llsMatrixNumRows() + constraintMatrixNumRows() >= constraintMatrixNumCols() )
                    && constraintMatrixNumRows() <= constraintMatrixNumCols() );
//                    assert( LLSMatrixNumCols_ == constraintMatrixNumCols_ );
//                    assert( (LLSMatrixNumRows_ + constraintMatrixNumRows_ >= constraintMatrixNumCols_) && (constraintMatrixNumRows_ <= constraintMatrixNumCols_) );

                    systemMatrix.resize( (LLSMatrixNumCols_ + constraintMatrixNumRows_), (LLSMatrixNumCols_ + constraintMatrixNumRows_), 0 );

//                    std::cout << "I reached here 1" << std::endl;

//                    std::cout << systemMatrix.cols() << std::endl;
//                    std::cout << systemMatrix.rows() << std::endl;

                    std::cout << "System matrix: " << std::endl;
                    printMatrix(systemMatrix);

//                    std::cout << "I reached here 2" << std::endl;

//                    std::cout << LLSMatrixNumCols_ << std::endl;

                    // fill up system matrix
                    for ( Size i = 0; i < systemMatrix.rows(); ++i) {
//                        std::cout << "I reached here 3" << std::endl;
                        if (i < LLSMatrixNumCols_)
                        {
//                            std::cout << "I reached here 4" << std::endl;
                            for (Size j = 0; j < systemMatrix.cols(); ++j){
                                if (j < LLSMatrixNumCols_)
                                {
//                                    std::cout << "I reached here 5" << std::endl;

                                    for (Size k = 0; k < LLSMatrixNumRows_; ++k)
                                        systemMatrix[ i ][ j ] += 2 * llsMatrix_[k][i] * llsMatrix_[k][j];
                                }
                                else
                                {
//                                    std::cout << "I reached here 6" << std::endl;
                                    systemMatrix[ i ][ j ] += constraintMatrix_[ j - LLSMatrixNumCols_ ][ i ];
                                }
                            }
                        }

                        else {
                            for (Size j = 0; j < LLSMatrixNumCols_; ++j) {
//                                std::cout << "I reached here 7 " << "i: " << i << " " << constraintMatrixNumRows_
//                                          << std::endl;
//                                int m = i - constraintMatrixNumRows();
//
//
//                                std::cout << (i - constraintMatrixNumRows_) << std::endl;
//                                std::cout << i - constraintMatrixNumRows() << std::endl;
//                                std::cout << m << std::endl;

                                systemMatrix[i][j] += constraintMatrix_[i - constraintMatrixNumRows_][j];
//                                std::cout << "I reached here 8" << std::endl;
                            }
                        }

//                        std::cout << "I reached here 9" << std::endl;
                    }

//                    for ( Size i = 0; i < LLSMatrixNumCols_; ++i)
//                        for ( Size j = 0; j < LLSMatrixNumCols_; ++j)
//                            for ( Size k = 0; k < LLSMatrixNumRows_; ++k)
//                                systemMatrix[ i ][ j ] += 2*llsMatrix_[ k ][ i ]*llsMatrix_[ k ][ j ];

                    std::cout << "System matrix: " << std::endl;
                    printMatrix(systemMatrix);

//                    std::cout << systemMatrix.size() << std::endl;

                    assert(LLSMatrixNumCols_+1 < systemMatrix.size());
//
//                    for ( int i = (LLSMatrixNumCols_ + 1 ); i < systemMatrix.size(); ++i)
//                        for ( int j = 0; j < LLSMatrixNumCols_; ++j)
//                            systemMatrix[ i ][ j ] += constraintMatrix_[ i ][ j ];

//                    std::cout << "I reached here 3" << std::endl;
//                    printMatrix(systemMatrix);

//                    for ( Size i = 0; i < LLSMatrixNumCols_; ++i)
//                        for ( Size j = (LLSMatrixNumCols_ +1); j < systemMatrix.size(); ++j)
//                            systemMatrix[ i ][ j ] += constraintMatrix_[ j ][ i ];
//
//                    printMatrix(systemMatrix);

//                    Vector systemVector = vectorSetUp()
//                    systemMatrix.solve()

                    systemMatrix.invert();

                    std::cout << "System matrix invert: " <<std::endl;
                    printMatrix(systemMatrix);

                    return systemMatrix;
                }

                template <class Vector>
                Vector vectorSetUp( const Vector &b, const Vector &d )
                {
                    const Size LLSMatrixNumRows_ = llsMatrixNumRows();
                    const Size LLSMatrixNumCols_ = llsMatrixNumCols();
                    const Size constraintMatrixNumCols_ = constraintMatrixNumCols();
                    const Size constraintMatrixNumRows_ = constraintMatrixNumRows();

                    assert( (LLSMatrixNumRows_ == b.size()) && ((LLSMatrixNumCols_ + d.size())
                    == (LLSMatrixNumCols_ + constraintMatrixNumRows_ )));

//                    std::vector< typename Vector::value_t > systemVector( (LLSMatrixNumCols_ + constraintMatrixNumRows_), 0 );

                    Vector systemVector(LLSMatrixNumCols_ + constraintMatrixNumRows_);
                    Vector y(LLSMatrixNumCols_,0);

                    // calculate y = 2 * A^T * b
                    llsMatrix_.usmtv(2,b,y);

                    for (Size i = 0; i < systemVector.size(); ++i){
                        if (i < LLSMatrixNumCols_)
                        {
                            systemVector[i] = y[i];
//                            for(Size j = 0; j < LLSMatrixNumRows_; ++j)
//                            {
//                                systemVector[i] += 2*llsMatrix_[j][i]*b[j];
//                            }
                        }
                        else
                        {
                            systemVector[i] = d[i - y.size()];
                        }
                    }

                    std::cout << "System Vector: " << std::endl;
                    printVector(systemVector);

                    return systemVector;
                }

        };

    template <class Matrix>
        LeastSquares<Matrix> leastSquares(const Matrix &llsMatrix, const Matrix &constraintMatrix)
        { return LeastSquares<Matrix>(llsMatrix,constraintMatrix); }
    } // namespace Vem
} // namespace Dune

#endif
