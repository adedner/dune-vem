#ifndef DUNE_VEM_AGGLOMERATION_BOUNDINGBOX_HH
#define DUNE_VEM_AGGLOMERATION_BOUNDINGBOX_HH

#include <cassert>
#include <cstddef>

#include <array>
#include <utility>
#include <vector>

#include <dune/common/fvector.hh>

#include <dune/vem/agglomeration/agglomeration.hh>

namespace Dune
{

  namespace Vem
  {

    // BoundingBox
    // -----------

    template< class GridPart >
    using BoundingBox = std::pair< FieldVector< typename GridPart::ctype, GridPart::dimensionworld >, FieldVector< typename GridPart::ctype, GridPart::dimensionworld > >;



    // agglomerateBoundingBoxes
    // ------------------------

    template< class GridPart >
    inline static std::vector< BoundingBox< GridPart > > boundingBoxes ( const Agglomeration< GridPart > &agglomeration )
    {
      typedef typename GridPart::template Codim< 0 >::GeometryType GeometryType;

      BoundingBox< GridPart > emptyBox;
      for( int k = 0; k < GridPart::dimensionworld; ++k )
      {
        emptyBox.first[ k ] = std::numeric_limits< typename GridPart::ctype >::max();
        emptyBox.second[ k ] = std::numeric_limits< typename GridPart::ctype >::min();
      }

      std::vector< BoundingBox< GridPart > > boundingBoxes( agglomeration.size(), emptyBox );
      for( const auto element : elements( agglomeration.gridPart(), Partitions::interiorBorder ) )
      {
        BoundingBox< GridPart > &bbox = boundingBoxes[ agglomeration.index( element ) ];
        const GeometryType geometry = element.geometry();
        for( int i = 0; i < geometry.corners(); ++i )
        {
          const typename GeometryType::GlobalCoordinate corner = geometry.corner( i );
          for( int k = 0; k < GridPart::dimensionworld; ++k )
          {
            bbox.first[ k ] = std::min( bbox.first[ k ], corner[ k ] );
            bbox.second[ k ] = std::max( bbox.second[ k ], corner[ k ] );
          }
        }
      }

      return std::move( boundingBoxes );
    }

  } // namespace Vem

} // namespace Dune

#endif // #define DUNE_VEM_AGGLOMERATION_BOUNDINGBOX_HH
