#ifndef DUNE_VEM_MISC_LEASTSQUARES_HH
#define DUNE_VEM_MISC_LEASTSQUARES_HH

#include <vector>

#include <dune/common/dynmatrix.hh>
#include <dune/vem/misc/pseudoinverse.hh>

#include <assert.h>

namespace Dune {

    namespace Vem {

        template <class Matrix>
        struct ColumnVector
        {
            ColumnVector(Matrix &matrix, int col)
                    : matrix_(matrix), col_(col) {}
            int size() const { return matrix_.rows(); }
            // typename Matrix::value_type& operator[](int i) {return matrix_[i][col_];}
            template <class Vector>
            ColumnVector &operator=(const Vector &v)
            {
                std::cout << "v size: " << v.size()
                          <<  " size: "  << size() << std::endl;

                assert( v.size() == size() );
                for ( std::size_t i = 0; i < size(); ++i)
                    matrix_[i][col_] = v[i];
            }
            Matrix &matrix_;
            int col_;
        };
        template <class Matrix>
        ColumnVector<Matrix> columnVector(Matrix &matrix, int col)
        { return ColumnVector(matrix,col); }

        template< class Matrix>
        class LeastSquares {
            public:
                typedef typename Matrix::size_type Size;
                typedef typename Matrix::value_type Field;

//                template < class Field >
                LeastSquares(const Matrix &llsMatrix, const Matrix &constraintMatrix)
                : llsMatrix_(llsMatrix), constraintMatrix_(constraintMatrix),
                  systemMatrixInv_(matrixSetUp(llsMatrix_, constraintMatrix_))
                {
                }
                LeastSquares(const Matrix &llsMatrix)
                : llsMatrix_(llsMatrix), constraintMatrix_(emptyMatrix()), systemMatrixInv_(matrixSetUp(llsMatrix_))
                {
                }
                LeastSquares(const LeastSquares &source) = delete;

                template <class Vector>
                Vector solve(const Vector &b, const Vector &d){
                    assert( b.size() == llsMatrix_.rows() );
                    Vector systemMultiply, systemLagrange;

//                    std::cout << constraintMatrix_.size() <<  "is empty" << isEmpty(constraintMatrix_) << std::endl;

                    if ( isEmpty(constraintMatrix_) ){
                        systemMultiply.resize(llsMatrix_.cols());
                        systemMatrixInv_.mv(b,systemMultiply);
                    }
                    else {
//                        std::cout << "constraint rows" << constraintMatrix_.rows() << std::endl;
                        assert(d.size() == constraintMatrix_.rows());

                        Vector systemVector = vectorSetUp(b, d);

                        Size systemMatrixDim = systemMatrixInv_.rows();

                        // since systemMatrix square, rows = cols
                        systemLagrange.resize(systemMatrixDim,0);

                        for (Size i = 0; i < systemMatrixDim; ++i)
                            for (Size j = 0; j < systemMatrixDim; ++j)
                                systemLagrange[i] += systemMatrixInv_[i][j] * systemVector[j];

                        systemMultiply.resize(llsMatrix_.cols(),0);
                        // get rid of Lagrange multipliers
                        for( Size i = 0; i < systemMultiply.size(); ++i)
                            systemMultiply[i] = systemLagrange[i];
                    }
                    return systemMultiply;
                }

                template <class Vector>
                void solve(const std::vector< Vector > &bVec,
                           const std::vector< Vector > &dVec,
                           std::vector< Vector > &solnVec){
                    // check dimensions match
                    assert( (bVec.size() == dVec.size()) && (dVec.size() == solnVec.size()));

                    for ( unsigned int i = 0; i < bVec.size(); ++i ){
                        solnVec[i] = solve(bVec[i],dVec[i]);
                    }
                }

                void printMatrix(const Matrix &A)
                {
                  return;
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
                  return;
                    for(unsigned int i = 0; i < x.size(); ++i)
                        std::cout << x[i] << std::endl;
                }



            private:
                const Matrix &llsMatrix_;
                const Matrix constraintMatrix_;
                const Matrix systemMatrixInv_;

                Matrix emptyMatrix()
                {
                    // return matrix with no size
                    Matrix A;
//                    std::cout << "size of empty matrix " << A.size() << std::endl;
                    return A;
                }

                bool isEmpty(const Matrix &A)
                {
                    return ( A.size() == 0 );
                }

//                template <class Field>
                Matrix matrixSetUp(const Matrix &llsMatrix_)
                {
                    // !!! this needs changing
                    LeftPseudoInverse< Field > pseudoInverse( llsMatrix_.cols() );
//                    LeftPseudoInverse< Field > pseudoInverse( llsMatrix_.cols() );

                    // no constraints in this case and so form pseudo inverse
//                    std::cout << "Matrix C has no size" << std::endl;

                    Matrix llsMatrixPseudoInv( llsMatrix_.cols(), llsMatrix_.rows() );

                    pseudoInverse( llsMatrix_, llsMatrixPseudoInv);

//                    std::cout << "pseudo Inv of A " << std::endl;
//                    printMatrix(llsMatrixPseudoInv);
                    return llsMatrixPseudoInv;
                }

                Matrix matrixSetUp(const Matrix &llsMatrix_, const Matrix &constraintMatrix_)
                {
                    if ( isEmpty(constraintMatrix_) ) {
                        return matrixSetUp(llsMatrix_);
                    }
                    else {
                        // construct the matrix [2A^T*A C^T ; C 0] needed for least squares solution
                        Matrix systemMatrix;

                        // check dimensions compatible
                        assert(llsMatrix_.cols() == constraintMatrix_.cols());
                        assert((llsMatrix_.rows() + constraintMatrix_.rows() >= constraintMatrix_.cols())
                               && constraintMatrix_.rows() <= constraintMatrix_.cols());

                        systemMatrix.resize((llsMatrix_.cols() + constraintMatrix_.rows()),
                                            (llsMatrix_.cols() + constraintMatrix_.rows()), 0);

//                        std::cout << "System matrix: " << std::endl;
//                        printMatrix(systemMatrix);

                        // fill up system matrix
                        for (Size i = 0; i < systemMatrix.rows(); ++i) {
                            if (i < llsMatrix_.cols()) {
                                for (Size j = 0; j < systemMatrix.cols(); ++j) {
                                    if (j < llsMatrix_.cols()) {
                                        // fill upper left
                                        for (Size k = 0; k < llsMatrix_.rows(); ++k)
                                            systemMatrix[i][j] += 2 * llsMatrix_[k][i] * llsMatrix_[k][j];
                                    } else {
                                        // fill upper right
                                        systemMatrix[i][j] += constraintMatrix_[j - llsMatrix_.cols()][i];
                                    }
                                }
                            } else {
                                for (Size j = 0; j < llsMatrix_.cols(); ++j)
                                    systemMatrix[i][j] += constraintMatrix_[i - llsMatrix_.cols()][j];
                            }
                        }

//                        std::cout << "System matrix: " << std::endl;
//                        printMatrix(systemMatrix);

                        assert(llsMatrix_.cols() + 1 < systemMatrix.size());

                        systemMatrix.invert();

//                        std::cout << "System matrix invert: " << std::endl;
//                        printMatrix(systemMatrix);

                        return systemMatrix;
                    }
                }

                template <class Vector>
                Vector vectorSetUp( const Vector &b, const Vector &d )
                {
                    assert( ( llsMatrix_.rows() == b.size()) && (( llsMatrix_.cols() + d.size())
                    == (llsMatrix_.cols() + constraintMatrix_.rows() )));

                    Vector systemVector( llsMatrix_.cols() + constraintMatrix_.rows()), y(llsMatrix_.cols(),0);

                    // calculate y = 2 * A^T * b
                    llsMatrix_.usmtv(2,b,y);

                    for (Size i = 0; i < systemVector.size(); ++i){
                        if (i < llsMatrix_.cols() ) {
                            systemVector[i] = y[i];
                        }
                        else {
                            systemVector[i] = d[i - y.size()];
                        }
                    }

//                    std::cout << "System Vector: " << std::endl;
//                    printVector(systemVector);

                    return systemVector;
                }

        };

        template <class Matrix>
        LeastSquares<Matrix> leastSquares(const Matrix &llsMatrix, const Matrix &constraintMatrix)
        { return LeastSquares<Matrix>(llsMatrix,constraintMatrix); }

        template <class Matrix>
        LeastSquares<Matrix> leastSquares(const Matrix &llsMatrix)
        { return LeastSquares<Matrix>(llsMatrix); }
    } // namespace Vem
} // namespace Dune

#endif
