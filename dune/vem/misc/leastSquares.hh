#ifndef DUNE_VEM_MISC_LEASTSQUARES_HH
#define DUNE_VEM_MISC_LEASTSQUARES_HH

#include <vector>

#include <dune/common/dynmatrix.hh>
#include <dune/vem/misc/pseudoinverse.hh>

#include <assert.h>

//using namespace std;

namespace Dune {

    namespace Vem {

        template <class Matrix>
        struct ColumnVector
        {
            ColumnVector(Matrix &matrix, int col)
                    : matrix_(matrix), col_(col) {}
            int size() const { return matrix_.rows(); }
            typename Matrix::value_type& operator[](int i) {return matrix_[i][col_];}
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

//                template < class Field >
                LeastSquares(const Matrix &llsMatrix, const Matrix &constraintMatrix)
                : llsMatrix_(llsMatrix), constraintMatrix_(constraintMatrix),
                  systemMatrixInv_(matrixSetUp())
                {
                }
                LeastSquares(const LeastSquares &source) = delete;

                template <class Vector, class Solution>
                void solve(const Vector &b, const Vector &d, Solution &solution){
                    assert( b.size() == llsMatrix_.rows() );
                    assert( solution.size() == llsMatrix_.cols() );

                    Vector systemMultiply;

                    if ( constraintMatrix_.size() == 0 ){
                        systemMultiply.resize(llsMatrix_.cols());
                        systemMatrixInv_.mv(b,systemMultiply);
                    }
                    else {
                        assert(d.size() == constraintMatrix_.rows());

                        Vector systemVector = vectorSetUp(b, d);

                        Size systemMatrixDim = systemMatrixInv_.rows();

                        // since systemMatrix square, rows = cols
                        systemMultiply.resize(systemMatrixDim,0);

                        for (Size i = 0; i < systemMatrixDim; ++i)
                            for (Size j = 0; j < systemMatrixDim; ++j)
                                systemMultiply[i] += systemMatrixInv_[i][j] * systemVector[j];
                    }
                    for (Size i = 0; i < solution.size(); ++i)
                        solution[i] = systemMultiply[i];
                }

                template <class Vector>
                Vector solve(const Vector &b, const Vector &d ){
                    Vector solution(llsMatrix_.cols());
                    solve(b,d,solution);
                    return solution;
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

//                template <class Field>
                Matrix matrixSetUp()
                {
                    if ( constraintMatrix_.size() == 0) {
                        // this needs changing
                        LeftPseudoInverse< double > pseudoInverse( llsMatrix_.cols() );

                        // no constraints in this case and so form pseudo inverse
                        std::cout << "Matrix C has no size" << std::endl;

                        Matrix llsMatrixPseudoInv;
                        llsMatrixPseudoInv.resize( llsMatrix_.cols(), llsMatrix_.rows() );

                        pseudoInverse( llsMatrix_, llsMatrixPseudoInv);

                        std::cout << "pseudo Inv of A " << std::endl;
                        printMatrix(llsMatrixPseudoInv);

                        return llsMatrixPseudoInv;
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

                        std::cout << "System matrix: " << std::endl;
                        printMatrix(systemMatrix);

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

                        std::cout << "System matrix: " << std::endl;
                        printMatrix(systemMatrix);

                        assert(llsMatrix_.cols() + 1 < systemMatrix.size());

                        systemMatrix.invert();

                        std::cout << "System matrix invert: " << std::endl;
                        printMatrix(systemMatrix);

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
