#ifndef DUNE_VEM_MISC_LEASTSQUARES_HH
#define DUNE_VEM_MISC_LEASTSQUARES_HH

#include <vector>

#include <dune/common/dynmatrix.hh>
#include <dune/vem/misc/pseudoinverse.hh>

#include <assert.h>

namespace Dune {

    namespace Vem {

        // template <class Matrix = Dune::DynaicMatrix<double, Vector = Dune::DynamicVector<double>>>
        template<class Matrix, class CMatrix>
        class LeastSquares {
        public:
            typedef typename Matrix::size_type Size;
            typedef typename Matrix::value_type Field;

            LeastSquares(const Matrix &llsMatrix, const CMatrix &constraintMatrix)
                    : llsMatrix_(llsMatrix), constraintMatrix_(constraintMatrix),
                      systemMatrixInv_(matrixSetUp(llsMatrix_, constraintMatrix_)) {
            }

            LeastSquares(const Matrix &llsMatrix)
                    : llsMatrix_(llsMatrix), constraintMatrix_(), systemMatrixInv_(matrixSetUp(llsMatrix_)) {
            }

            LeastSquares(const LeastSquares &source) = delete;

            template<class bVector, class dVector>
            bVector solve(const bVector &b, const dVector &d) {

              // printVector(b);
              bVector systemMultiply, systemLagrange;

              if (isEmpty(constraintMatrix_)) { // this will usually require computing A^Tb
                assert(b.size() == llsMatrix_.rows());
                systemMultiply.resize(llsMatrix_.cols());
                systemMatrixInv_.mv(b, systemMultiply);
              }
              else if (isEmpty(llsMatrix_)) {
                assert(d.size() == constraintMatrix_.rows());
                systemMultiply.resize(constraintMatrix_.cols());
                systemMatrixInv_.mv(d, systemMultiply);
              } else {
                assert(b.size() == llsMatrix_.rows());
                assert(d.size() == constraintMatrix_.rows());

                auto systemVector = vectorSetUp(b, d);

                Size systemMatrixDim = systemMatrixInv_.rows();

                // since systemMatrix square, rows = cols
                systemLagrange.resize(systemMatrixDim, 0);

                for (Size i = 0; i < systemMatrixDim; ++i)
                  for (Size j = 0; j < systemMatrixDim; ++j)
                    systemLagrange[i] += systemMatrixInv_[i][j] * systemVector[j];

                systemMultiply.resize(llsMatrix_.cols(), 0);
                // get rid of Lagrange multipliers
                // TODO? avoid copy by cutting off in operator= of ColVec
                for (Size i = 0; i < systemMultiply.size(); ++i)
                  systemMultiply[i] = systemLagrange[i];
              }
              /*
              std::cout << "'''''''''''''''''\n";
              printVector(systemMultiply);
              std::cout << "'''''''''''''''''\n";
              */
              return systemMultiply;
            }

#if 0
            template<class Vector>
            void solve(const std::vector<Vector> &bVec,
                       const std::vector<Vector> &dVec,
                       std::vector<Vector> &solnVec) {
              // check dimensions match
              assert((bVec.size() == dVec.size()) && (dVec.size() == solnVec.size()));

              for (unsigned int i = 0; i < bVec.size(); ++i) {
                solnVec[i] = solve(bVec[i], dVec[i]);
              }
            }
#endif
            void printMatrix(const Matrix &A) {
              for (unsigned int i = 0; i < A.rows(); ++i) {
                for (unsigned int j = 0; j < A.cols(); ++j) {
                  std::cout << A[i][j] << " ";
                }
                std::cout << std::endl;
              }
            }

            template<class xVector>
            void printVector(const xVector &x) {
              for (unsigned int i = 0; i < x.size(); ++i)
              {
                std::cout << x[i] << " ";
              }
              std::cout << std::endl;
            }


        private:
            const Matrix &llsMatrix_;
            // TODO: avoid copy of constraintMatrix in constructor
            const CMatrix constraintMatrix_;
            const Matrix systemMatrixInv_;

            template<class xMatrix>
            bool isEmpty(const xMatrix &A) {
              return (A.size() == 0);
            }

            // return inverse of a matrix
            template <class CCMatrix>
            Matrix matrixInv(const CCMatrix &matrix) {
              // std::cout << "@@@@@@@@@@@@@@@@@\nVORHER\n";
              // printMatrix(matrix);
              Matrix inverse(matrix);
              inverse.invert();
              // std::cout << "@@@@@@@@@@@@@@@@@\nAFTER\n";
              // printMatrix(inverse);
              // std::cout << "@@@@@@@@@@@@@@@@@\n";
              return inverse;
            }
            // return pseudo inverse of a matrix
            template <class CCMatrix>
            Matrix matrixSetUp(const CCMatrix &matrix) {
              LeftPseudoInverse <Field> pseudoInverse(matrix.cols());
              Matrix matrixPseudoInv(matrix.cols(), matrix.rows());
              pseudoInverse(matrix, matrixPseudoInv);
              return matrixPseudoInv;
            }

            // TODO: avoid usage of '_' in parameter name - either use
            // static method or have no parameters
            template <class LMatrix,class CCMatrix>
            Matrix matrixSetUp(const LMatrix &llsMatrix_, const CCMatrix &constraintMatrix_) {
              if (isEmpty(constraintMatrix_)) {
                return matrixSetUp(llsMatrix_);
              }
              else if (isEmpty(llsMatrix_)) {
                // Andreas: here a normal inverse should work - this produces a pseudo inverse
                return matrixSetUp(constraintMatrix_);
              } else {
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

                // std::cout << "LLS+C @@@@@@@@@@@@@@@@@\nAFTER\n";
                // printMatrix(systemMatrix);
                // std::cout << "LLS+C @@@@@@@@@@@@@@@@@\n";
                return systemMatrix;
              }
            }

            template<class bVector, class dVector>
            bVector vectorSetUp(const bVector &b, const dVector &d) {
              assert((llsMatrix_.rows() == b.size()) && ((llsMatrix_.cols() + d.size())
                                                         == (llsMatrix_.cols() + constraintMatrix_.rows())));

              bVector systemVector(llsMatrix_.cols() + constraintMatrix_.rows()), y(llsMatrix_.cols(), 0);

              // calculate y = 2 * A^T * b
              llsMatrix_.usmtv(2, b, y);

              for (Size i = 0; i < systemVector.size(); ++i) {
                if (i < y.size()) {
                  systemVector[i] = y[i];
                } else {
                  systemVector[i] = d[i - y.size()];
                }
              }

//                    std::cout << "System Vector: " << std::endl;
//                    printVector(systemVector);

              return systemVector;
            }

        };

        template<class Matrix,class CMatrix>
        LeastSquares<Matrix,CMatrix> leastSquares(const Matrix &llsMatrix, const CMatrix &constraintMatrix) {
          return LeastSquares<Matrix,CMatrix>(llsMatrix, constraintMatrix);
        }

        template<class Matrix>
        LeastSquares<Matrix,Matrix> leastSquares(const Matrix &llsMatrix) { return LeastSquares<Matrix,Matrix>(llsMatrix); }
    } // namespace Vem
} // namespace Dune

#endif
