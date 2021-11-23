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
#include <dune/vem/misc/vector.hh>

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
      Agglomeration ( GridPartType &gridPart, bool rotate, const Callback callBack )
        : gridPart_( gridPart ),
          rotate_(rotate),
          mapper_( gridPart, mcmgElementLayout() ),
          indices_( mapper_.size() ),
          size_( 0 ),
          maxOrder_( 0 ),
          counter_( 0 )
      {
        const auto &is = gridPart.indexSet();
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
      ~Agglomeration()
      {
      }
      Agglomeration(const Agglomeration&) = delete;
      Agglomeration &operator=( Agglomeration&) = delete;

      std::size_t counter()
      {
        return counter_;
      }
      void update()
      {
        // std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        boundingBoxes_ = Dune::Vem::boundingBoxes( *this, rotate_ );
        if (maxOrder_>0)
          Dune::Vem::onbBasis(*this, maxOrder_, boundingBoxes());
        assert(boundingBoxes()->size() == size());
        ++counter_;
        /*
        auto end = std::chrono::system_clock::now();
        auto diff = std::chrono::duration_cast < std::chrono::seconds > (end - start).count();
        std::cout << "Total AgglUpdate = " << diff << " seconds for "
                  << size() << " BBs " << std::endl;
        */
      }
      void onbBasis(int order)
      {
        if (order > maxOrder_)
        {
          Dune::Vem::onbBasis(*this, order, boundingBoxes());
          maxOrder_ = order;
        }
      }

      GridPart &gridPart () const { return gridPart_; }

      std::size_t index ( const ElementType &element ) const { return indices_[ mapper_.index( element ) ]; }

      std::size_t size () const { return size_; }

      const BoundingBox<GridPart>& boundingBox( std::size_t index ) const
      {
        assert(index<boundingBoxes()->size());
        return (*boundingBoxes())[index];
      }
      const BoundingBox<GridPart>& boundingBox( const ElementType &element ) const
      {
        return boundingBox( index( element ) );
      }
      std::shared_ptr< Std::vector< BoundingBox< GridPart > > > boundingBoxes() const
      {
        return boundingBoxes_;
      }

    private:
      GridPart &gridPart_;
      bool rotate_;
      MultipleCodimMultipleGeomTypeMapper< typename GridPartType::GridViewType > mapper_;
      Std::vector< std::size_t > indices_;
      std::size_t size_;
      std::size_t maxOrder_;
      std::size_t counter_;
      std::shared_ptr< Std::vector< BoundingBox< GridPart > > > boundingBoxes_;
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
