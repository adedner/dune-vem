#ifndef DUNE_VEM_AGGLOMERATION_AGGLOMERATION_HH
#define DUNE_VEM_AGGLOMERATION_AGGLOMERATION_HH

#include <cassert>
#include <cstddef>

#include <algorithm>
#include <utility>
#include <vector>

#include <dune/grid/common/mcmgmapper.hh>
#include <dune/vem/agglomeration/boundingbox.hh>
#include <dune/vem/agglomeration/basisfunctionset.hh>

namespace Dune
{

  namespace Vem
  {

    // Agglomeration
    // -------------

    template< class GridPart >
    class Agglomeration
    {
      typedef Agglomeration< GridPart > ThisType;

    public:
      typedef GridPart GridPartType;

      typedef typename GridPartType::template Codim< 0 >::EntityType ElementType;

      template <class Callback>
      Agglomeration ( GridPartType &gridPart, const Callback callBack )
        : gridPart_( gridPart ),
          mapper_( static_cast< typename GridPartType::GridViewType >( gridPart ), mcmgElementLayout() ),
          indices_( mapper_.size() ),
          size_( 0 )
      {
        auto is = gridPart.indexSet();
        const auto &end = gridPart.template end<0>();
        for ( auto it = gridPart.template begin<0>(); it != end; ++it )
        {
          const auto &element = *it;
          indices_[ mapper_.index( element ) ] = callBack( is.index(element) );
        }
        assert( indices_.size() == static_cast< std::size_t >( mapper_.size() ) );
        if( !indices_.empty() )
          size_ = *std::max_element( indices_.begin(), indices_.end() ) + 1u;

        update();
      }
      ~Agglomeration() {}
      Agglomeration(const Agglomeration&) = delete;
      Agglomeration &operator=( Agglomeration&) = delete;

      void update()
      {
        boundingBoxes_ = Dune::Vem::boundingBoxes( *this );
      }
      void onbBasis(int order)
      {
        Dune::Vem::onbBasis(*this, order, boundingBoxes());
      }

      GridPart &gridPart () const { return gridPart_; }

      std::size_t index ( const ElementType &element ) const { return indices_[ mapper_.index( element ) ]; }

      std::size_t size () const { return size_; }

      const BoundingBox<GridPart>& boundingBox( std::size_t index ) const
      {
        assert(index<boundingBoxes_.size());
        return boundingBoxes_[index];
      }
      const BoundingBox<GridPart>& boundingBox( const ElementType &element ) const
      {
        return boundingBox( index( element ) );
      }
      const std::vector< BoundingBox< GridPart > >& boundingBoxes() const
      {
        return boundingBoxes_;
      }
      std::vector< BoundingBox< GridPart > >& boundingBoxes()
      {
        return boundingBoxes_;
      }

    private:
      template< class T >
      static std::vector< std::size_t > convert ( const std::vector< T > &v )
      {
        std::vector< std::size_t > w;
        w.reserve( v.size() );
        for( const T &i : v )
          w.emplace_back( i );
        return w;
      }

      GridPart &gridPart_;
      MultipleCodimMultipleGeomTypeMapper< typename GridPartType::GridViewType > mapper_;
      std::vector< std::size_t > indices_;
      std::size_t size_;
      std::vector< BoundingBox< GridPart > > boundingBoxes_;
    };



    // LocalAgglomerationFunction
    // --------------------------

    template< class GridPart >
    struct LocalAgglomerationFunction
    {
      typedef typename Agglomeration< GridPart >::ElementType Entity;

      explicit LocalAgglomerationFunction ( const Agglomeration< GridPart > &agglomeration ) : agglomeration_( agglomeration ) {}

      std::size_t operator() ( const typename Entity::Geometry::LocalCoordinate & ) const
      {
        assert( entity_ );
        return agglomeration_.index( *entity_ );
      }

      void bind ( const Entity &entity ) { entity_ = &entity; }
      void unbind () { entity_ = nullptr; }

    private:
      const Agglomeration< GridPart > &agglomeration_;
      const Entity *entity_ = nullptr;
    };



    // localFunction for Agglomeration
    // -------------------------------

    template< class GridPart >
    inline static LocalAgglomerationFunction< GridPart > localFunction ( const Agglomeration< GridPart > &agglomeration )
    {
      return LocalAgglomerationFunction< GridPart >( agglomeration );
    }

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_AGGLOMERATION_HH
