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
                  systemMatrixiInv_(matrixSetUp())
                {
                }
                LeastSquares(const LeastSquares &source) = delete;

                template <class Vector>
                Vector leastSquaresSolver(const Vector &b, const Vector &d ){

                    assert( (b.size() == d.size()) );

                    Vector solution(b.size());
                    Vector systemVector = vectorSetUp(b,d);

                    Size systemMatrixDim = systemMatrix_.rows();

                    // since systemMatrix sqaure, rows = cols
                    Vector systemMultiply(systemMatrixDim, 0);


                    for ( Size i = 0; i < systemMatrixDim; ++i) {
                        for (Size j = 0; j < systemMatrixDim; ++j)
                            systemMultiply[i] += systemMatrix[i][j] * systemVector[j];
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


            private:
                const Matrix &LLSMatrix_;
                const Matrix &constraintMatrix_;
                const Matrix systemMatrixInv_;
                Size LLSMatrixNumRows_ = LLSMatrix_.rows();
                Size LLSMatrixNumCols_ = LLSMatrix_.cols();
                Size constraintMatrixNumRows_ = constraintMatrix_.rows();
                Size constraintMatrixNumCols_ = constraintMatrix_.cols();

                Matrix matrixSetUp()
                {
                    // construct the matrix [2A^T*A C^T ; C 0] needed for least squares solution
                    Matrix systemMatrix;

                    // check dimensions compatible
                    assert( LLSMatrixNumCols_ == constraintMatrixNumCols_ );
                    assert( (LLSMatrixNumRows_ + constraintMatrixNumRows_ >= constraintMatrixNumCols_) && (constraintMatrixNumRows_ <= constraintMatrixNumCols_) );

                    systemMatrix.resize( (LLSMatrixNumCols_ + constraintMatrixNumRows_), (LLSMatrixNumCols_ + constraintMatrixNumRows_), 0 );

                    // fill up system matrix
                    for ( Size i = 0; i < LLSMatrixNumCols_; ++i)
                        for ( Size j = 0; j < LLSMatrixNumCols_; ++j)
                            for ( Size k = 0; k < LLSMatrixNumRows_; ++k)
                                systemMatrix[ i ][ j ] += 2*LLSMatrix_[ k ][ i ]*LLSMatrix_[ k ][ j ];

                    for ( Size i = (LLSMatrixNumCols_ + 1 ); i < systemMatrix.size(); ++i)
                        for ( Size j = 0; j < LLSMatrixNumCols_; ++j)
                            systemMatrix[ i ][ j ] += constraintMatrix_[ i ][ j ];

                    for ( Size i = 0; i < LLSMatrixNumCols_; ++i)
                        for ( Size j = (LLSMatrixNumCols_ +1); j < systemMatrix.size(); ++j)
                            systemMatrix[ i ][ j ] += constraintMatrix_[ j ][ i ];

                    systemMatrix.invert();
                    return systemMatrix;
                }

                template <class Vector>
                Vector vectorSetUp( const Vector &b, const Vector &d )
                {
                    assert( (LLSMatrixNumRows_ == b.size()) && ((LLSMatrixNumCols_ + d.size())
                    == (LLSMatrixNumCols_ + constraintMatrixNumRows_ )));

                    std::vector< Field > systemVector( (LLSMatrixNumCols_ + constraintMatrixNumRows_), 0 );

                    for (Size i = 0; i < systemVector.size(); ++i){
                        if (i < LLSMatrixNumCols_)
                        {

                            for(Size j = 0; j < LLSMatrixNumRows_; ++j)
                            {
                                systemVector[i] += 2*LLSMatrix_[j][i]*b[j];
                            }
                        }
                        else
                        {
                            systemVector[i] += d[i];
                        }
                    }
                    return systemVector;
                }

        };

    template <class Matrix>
        LeastSquares<Matrix> leastSquares(const Matrix &llsMatrix, const Matrix &constraintMatrix) { return LeastSquares<Matrix>(llsMatrix,constrainedMatrix); }
    } // namespace Vem
} // namespace Dune

#endif
