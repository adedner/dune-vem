#ifndef DUNE_VEM_MISC_LEASTSQUARES_HH
#define DUNE_VEM_MISC_LEASTSQUARES_HH

#include <vector>

#include <dune/common/dynmatrix.hh>
#include <dune/vem/misc/pseudoinverse.hh>

#include <assert.h>

namespace Dune {

    namespace Vem {

//        template< class Matrix >
//        struct expandColumnVector
//        {
//            expandColumnVector(Matrix &matrix, int col)
//                    : matrix_(matrix), col_(col) {}
//
//            template <class Vector>
//            Vector expand(){
//              Vector v(2*matrix_.size(),0);
//              for ( std::size_t i = 0; i < 2*matrix_.size(); ++i)
//                if ( i < matrix_.size() )
//                  v[i] = matrix_[i][col_][0];
//                else
//                  v[i] = matrix_[ i - matrix_.size() ][col_][1];
//            }
//            Matrix &matrix_;
//            int col_;
//        };
//        template < class Matrix >
//        expandColumnVector<Matrix> expandColumnVector(Matrix &matrix, int col)
//        { return expandColumnVector(matrix,col);}

        // C -> (C 0 ... 0)
        //      (0 C ... 0)
        //      ( ....... )
        //      (0 ... 0 C)
        template < class Matrix >
        struct BlockMatrix
        {
          struct RowWrapper
          {
            RowWrapper(const Matrix &matrix,int block, int row)
            : matrix_(matrix), block_(block), row_(row) {}
            typename Matrix::value_type operator[](int col) const {
              /*
              for (int i=0;i<block_;++i)
                if (row < (i+1)*matrix_.rows() && col < (i+1)*matrix_.cols())
                  return matrix_[i*matrix_.rows()+row][i*matrix_.cols()+col];
              */
              int c = col/block_;
              int r = row_/block_;
              assert(c<matrix_.cols() && r<matrix_.rows());
              if (col%block_ == row_%block_)
                return matrix_[r][c];
              return 0;
            }
            const Matrix &matrix_;
            int block_, row_;
          };
          BlockMatrix(const Matrix &matrix, int block)
            : matrix_(matrix), block_(block) {}
          const RowWrapper operator[](int row) const {
            return RowWrapper(matrix_,block_,row);
          }
          unsigned int rows() const {return matrix_.rows()*block_;}
          unsigned int cols() const {return matrix_.cols()*block_;}

          const Matrix &matrix_;
          int block_;
        };
        template <class Matrix >
        BlockMatrix<Matrix> blockMatrix(const Matrix &matrix, int block)
        { return BlockMatrix<Matrix>(matrix,block); }


        template <class F>
        struct VectorizedF
        {
          static const int size = 1;
          static const F &get(const F &x, int rows, int r)
          { return x; }
          template <class V>
          static void assign(F &in, const V &v, int i)
          { in = v[i]; }
        };
        template <class F, int N>
        struct VectorizedF<Dune::FieldVector<F,N>>
        {
          static const int size = N;
          static const F get(const Dune::FieldVector<F,N> &x, int rows, int r)
          { return x[r/rows]; }
          template <class V>
          static void assign(Dune::FieldVector<F,N> &in, const V &v, int i)
          { for (int r=0;r<N;++r) in[r] = v[i+r*v.size()/N]; }
        };
        template <class F, int R, int C>
        struct VectorizedF<Dune::FieldMatrix<F,R,C>>
        {
          static const int size = R*C;
          static const F &get(const Dune::FieldMatrix<F,R,C> &x, int rows, int r)
          { return x[r/(R*rows)][r/rows%C]; }
        };
        template <class FStruct>
        struct VectorizedF<std::vector<FStruct>>
        : public VectorizedF<FStruct> {};

        // take a column of a matrix with value_type V and make that
        // column into one long vector
        template < class Matrix >
        struct VectorizeMatrixCol
        {
          typedef VectorizedF<typename Matrix::value_type> VF;
          VectorizeMatrixCol(Matrix &matrix, int col)
            : matrix_(matrix), col_(col) {}
          unsigned int size() const {return matrix_.size()*VF::size;}
          const typename Matrix::value_type &operator[](int row) const
          {
            return VF::get(matrix_[row%matrix_.rows()][col_], matrix_.rows(),row);
          }
          template <class Vector>
          VectorizeMatrixCol &operator=(const Vector &v)
          {
            assert( v.size() == size() );
            for ( std::size_t i = 0; i < matrix_.size(); ++i)
              VF::assign(matrix_[i][col_], v, i);
            return *this;
          }

          Matrix &matrix_;
          int col_;
        };
        template <class Matrix>
        VectorizeMatrixCol<Matrix> vectorizeMatrixCol(Matrix &matrix, int col)
        { return VectorizeMatrixCol(matrix,col); }

        template< class Matrix>
        class LeastSquares {
            public:
                typedef typename Matrix::size_type Size;
                typedef typename Matrix::value_type Field;

                LeastSquares(const Matrix &llsMatrix, const Matrix &constraintMatrix)
                : llsMatrix_(llsMatrix), constraintMatrix_(constraintMatrix),
                  systemMatrixInv_(matrixSetUp(llsMatrix_, constraintMatrix_))
                {
                }
                LeastSquares(const Matrix &llsMatrix)
                : llsMatrix_(llsMatrix), constraintMatrix_(), systemMatrixInv_(matrixSetUp(llsMatrix_))
                {
                }
                LeastSquares(const LeastSquares &source) = delete;

                template <class Vector>
                Vector solve(const Vector &b, const Vector &d){

                  Vector systemMultiply, systemLagrange;

                  if ( isEmpty(constraintMatrix_) ){
                    assert( b.size() == llsMatrix_.rows() );
                    systemMultiply.resize(llsMatrix_.cols());
                        systemMatrixInv_.mv(b,systemMultiply);
                    }
                    if ( isEmpty(llsMatrix_) ){
                      assert(d.size() == constraintMatrix_.rows());
                      systemMultiply.resize(constraintMatrix_.cols());
                      systemMatrixInv_.mv(d,systemMultiply);
                    }
                    else {
                      assert( b.size() == llsMatrix_.rows() );
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
                      // TODO? avoid copy by cutting of in operator= of ColVec
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
                // TODO: avoid copy of constraintMatrix in constructor
                const Matrix constraintMatrix_;
                const Matrix systemMatrixInv_;

                bool isEmpty(const Matrix &A)
                {
                    return ( A.size() == 0 );
                }

                // return pseudo inverse of a matrix
                Matrix matrixSetUp(const Matrix &matrix)
                {
                    LeftPseudoInverse< Field > pseudoInverse( matrix.cols() );

                    Matrix matrixPseudoInv( matrix.cols(), matrix.rows() );

                    pseudoInverse( matrix, matrixPseudoInv);

                    return matrixPseudoInv;
                }

                // TODO: avoid usage of '_' in parameter name - either use
                // static method or have no parameters
                Matrix matrixSetUp(const Matrix &llsMatrix_, const Matrix &constraintMatrix_)
                {
                    if ( isEmpty(constraintMatrix_) ) {
                        return matrixSetUp(llsMatrix_);
                    }
                    if ( isEmpty(llsMatrix_) ) {
                      return matrixSetUp(constraintMatrix_);
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
                        if (i < y.size() ) {
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
