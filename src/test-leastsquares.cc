#include <dune/vem/misc/leastSquares.hh>

#include <dune/common/dynvector.hh>
#include <dune/common/dynmatrix.hh>

namespace Dune
{
    namespace Vem
    {
        int main()
        {
            DynamicMatrix< DomainFieldType > A, C;
            DynamicVector< DomainFieldType > b, d;
            DynamicVector< DomainFieldType > exactSoln;

            // define class member of least squares
            LeastSquares LeastSquaresMinimiser;

            LeastSquares(A,C);

            // invoke least squares solver
            DynamicVector< DomainFieldType > llsSolnVec;

            // compare with exact solution
            double error;
            // norm llsSolnVec - exactSoln;

            // define tolerance
            if (error < tol)
            {
                return 0;
            }

        }
    }
}
