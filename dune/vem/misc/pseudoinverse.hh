#ifndef DUNE_VEM_MISC_PSEUDOINVERSE_HH
#define DUNE_VEM_MISC_PSEUDOINVERSE_HH

#include <vector>

#include <dune/common/dynmatrix.hh>

namespace Dune
{

  namespace Vem
  {

    // PseudoInverse
    // -------------

    template< class Field >
    struct LeftPseudoInverse
    {
      typedef DynamicMatrix< Field > Matrix;

      typedef typename Matrix::size_type Size;

      explicit LeftPseudoInverse ( Size numCols ) : ATA_( numCols, numCols ) {}

      template< class AMatrix, class InvA >
      void operator() ( const AMatrix &A, InvA &invA )
      {
        Size numRows = A.rows();
        Size numCols = A.cols();

        assert( (numCols == ATA_.rows()) && (numCols == ATA_.cols()) );

        for( Size i = 0; i < numCols; ++i )
        {
          for( Size j = 0; j < numCols; ++j )
            ATA_[ i ][ j ] = 0;
          for( Size k = 0; k < numRows; ++k )
            for( Size j = 0; j < numCols; ++j )
              ATA_[ i ][ j ] += A[ k ][ i ] * A[ k ][ j ];
        }
        ATA_.invert();

        resize( invA, numCols, numRows );
        for( Size i = 0; i < numCols; ++i )
          for( Size j = 0; j < numRows; ++j )
          {
            invA[i][j] = 0;
            for( Size k = 0; k < numCols; ++k )
              invA[ i ][ j ] += ATA_[ i ][k] * A[ j ][k];
          }
      }


    private:
      void resize ( Matrix &A, Size numRows, Size numCols )
      {
        if( (A.rows() != numRows) || (A.cols() != numCols) )
          A.resize( numRows, numCols );
      }

      void resize ( std::vector< std::vector< Field > > &A, Size numRows, Size numCols )
      {
        A.resize( numRows );
        for( auto &row : A )
          row.resize( numCols );
      }

      Matrix ATA_;
    };

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_MISC_PSEUDOINVERSE_HH
