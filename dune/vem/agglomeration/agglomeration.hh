#ifndef DUNE_VEM_AGGLOMERATION_AGGLOMERATION_HH
#define DUNE_VEM_AGGLOMERATION_AGGLOMERATION_HH

#include <cassert>
#include <cstddef>

#include <algorithm>
#include <utility>
#include <vector>

#include <dune/grid/common/mcmgmapper.hh>

namespace Dune
{

  namespace Vem
  {

    // Agglomeration
    // -------------

    template< class GridPart >
    class Agglomeration
    {
      typedef Agglomeration< GridPart > This;

    public:
      typedef GridPart GridPartType;

      typedef typename GridPartType::template Codim< 0 >::EntityType ElementType;

      Agglomeration ( const GridPartType &gridPart, std::vector< std::size_t > indices )
        : gridPart_( gridPart ), mapper_( gridPart ), indices_( std::move( indices ) ), size_( 0 )
      {
        assert( indices_.size() == mapper.size() );
        if( !indices_.empty() )
          size_ = *std::max_element( agglomerateIndices.begin(), agglomerateIndices.end() ) + 1u;
      }

      const GridPart &gridPart () const { return gridPart_; }

      std::size_t index ( const ElementType &element ) const { return indices_[ mapper_.index( element ) ]; }

      std::size_t size () const { return size_; }

    private:
      const GridPart &gridPart_;
      MultipleCodimMultipleGeomTypeMapper< typename GridPartType::GridViewType, MCMGElementLayout > mapper_;
      std::vector< std::size_t > indices_;
      std::size_t size_;
    };

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_AGGLOMERATION_HH
