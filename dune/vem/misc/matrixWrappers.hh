#ifndef DUNE_VEM_MISC_MATRIXWRAPPERS_HH
#define DUNE_VEM_MISC_MATRIXWRAPPERS_HH

#include <vector>

#include <dune/common/dynmatrix.hh>
#include <dune/vem/misc/pseudoinverse.hh>

#include <assert.h>
#include <dune/vem/misc/vector.hh>


namespace Dune {

    namespace Vem {

        template<class F>
        struct VectorizedF {
            static const int size = 1;
            typedef F field_type;

            static const F &get(const F &x, int rows, int r) { return x; }

            template<class V>
            static void assign(F &in, const V &v, int i) { in = v[i]; }
        };

        template<class F, int N>
        struct VectorizedF<Dune::FieldVector<F, N>> {
            static const int size = N;
            typedef F field_type;

            static const F &get(const Dune::FieldVector<F, N> &x, int rows, int r)
            { return x[r / rows]; }

            template<class V>
            static void assign(Dune::FieldVector<F, N> &in, const V &v, int i) {
              for (int r = 0; r < N; ++r)
                in[r] = v[i + r * v.size() / N];
            }
        };

        template<class F, int R, int C>
        struct VectorizedF<Dune::FieldMatrix<F, R, C>> {
            static const int size = R * C;
            typedef F field_type;

            static const F &get(const Dune::FieldMatrix<F, R, C> &x, int rows, int r) {
              return x[r / (R * rows)][r / rows % C];
            }
        };

        template<class FStruct>
        struct VectorizedF<Std::vector<FStruct>>
                : public VectorizedF<FStruct> {
        };

        // take a column of a matrix with value_type V and make that
        // column into one long vector
        template<class Matrix>
        struct VectorizeMatrixCol {
            typedef VectorizedF<typename Matrix::value_type> VF;

            VectorizeMatrixCol(Matrix &matrix, int col)
                    : matrix_(matrix), col_(col) {}

            unsigned int size() const { return matrix_.size() * VF::size; }

            const typename VF::field_type &operator[](int row) const
            {
              return VF::get(matrix_[row % matrix_.rows()][col_], matrix_.rows(), row);
            }

            template<class Vector>
            VectorizeMatrixCol &operator=(const Vector &v) {
              assert(v.size() <= VF::size * size());
              for (std::size_t i = 0; i < v.size()/VF::size; ++i)
                VF::assign(matrix_[i][col_], v, i);
              return *this;
            }

            Matrix &matrix_;
            int col_;
        };

        template<class Matrix>
        VectorizeMatrixCol<Matrix> vectorizeMatrixCol(Matrix &matrix, int col) {
          return VectorizeMatrixCol(matrix, col);
        }

        // C -> (C 0 ... 0)
        //      (0 C ... 0)
        //      ( ....... )
        //      (0 ... 0 C)
        template<class Matrix>
        struct BlockMatrix
        {
          struct RowWrapper
          {
            RowWrapper(const Matrix &matrix, int block, int row)
            : matrix_(matrix), block_(block), row_(row) {}

            typename Matrix::value_type operator[](int col) const
            {
              /*
              for (int i=0;i<block_;++i)
                if (row < (i+1)*matrix_.rows() && col < (i+1)*matrix_.cols())
                  return matrix_[i*matrix_.rows()+row][i*matrix_.cols()+col];
              */
              int c = col  % matrix_.cols();
              int r = row_ % matrix_.rows();
              assert(c < matrix_.cols() && r < matrix_.rows());
              if (col / matrix_.cols() == row_ / matrix_.rows())
                return matrix_[r][c];
              return 0;
            }

            const Matrix &matrix_;
            int block_, row_;
          };

          BlockMatrix(const Matrix &matrix, int block)
          : matrix_(matrix), block_(block) {}

          const RowWrapper operator[](int row) const {
            return RowWrapper(matrix_, block_, row);
          }

          unsigned int size() const { return matrix_.rows() * block_; }
          unsigned int rows() const { return matrix_.rows() * block_; }
          unsigned int cols() const { return matrix_.cols() * block_; }

          const Matrix &matrix_;
          int block_;
        };
        template<class Matrix>
        BlockMatrix<Matrix> blockMatrix(const Matrix &matrix, int block) {
          return BlockMatrix<Matrix>(matrix, block);
        }

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
    }
}
#endif
