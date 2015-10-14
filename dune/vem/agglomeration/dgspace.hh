#ifndef DUNE_VEM_AGGLOMERATION_DGSPACE_HH
#define DUNE_VEM_AGGLOMERATION_DGSPACE_HH

#include <utility>

#include <dune/common/power.hh>

#include <dune/fem/space/common/commoperations.hh>
#include <dune/fem/space/common/defaultcommhandler.hh>
#include <dune/fem/space/common/functionspace.hh>
#include <dune/fem/space/shapefunctionset/legendre.hh>
#include <dune/fem/space/shapefunctionset/proxy.hh>
#include <dune/fem/space/shapefunctionset/vectorial.hh>

#include <dune/vem/agglomeration/basisfunctionset.hh>
#include <dune/vem/agglomeration/dgmapper.hh>

namespace Dune
{

  namespace Vem
  {

    // Internal Forward Declarations
    // -----------------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    class AgglomerateDGSpace;



    // AgglomerateDGSpaceTraits
    // ------------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    struct AgglomerateDGSpaceTraits
    {
      friend class AgglomerateDGSpace< FunctionSpace, GridPart, polOrder >;

      typedef AgglomerateDGSpace< FunctionSpace, GridPart, polOrder > DiscreteFunctionSpaceType;

      typedef FunctionSpace FunctionSpaceType;
      typedef GridPart GridPartType;

    private:
      typedef typename GridPartType::template Codim< 0 >::EntityType EntityType;

      typedef typename Fem::FunctionSpace< typename FunctionSpaceType::DomainFieldType, typename FunctionSpaceType::RangeFieldType, FunctionSpaceType::dimDomain, 1 > ScalarFunctionSpaceType;
      typedef Fem::LegendreShapeFunctionSet< ScalarFunctionSpaceType > ScalarShapeFunctionSetType;
      typedef Fem::VectorialShapeFunctionSet< Fem::ShapeFunctionSetProxy< ScalarShapeFunctionSetType >, typename FunctionSpaceType::RangeType > ShapeFunctionSetType;

    public:
      typedef BoundingBoxShapeFunctionSet< EntityType, ShapeFunctionSetType > BasisFunctionSetType;

      static const std::size_t localBlockSize = FunctionSpaceType::dimRange * StaticPower< polOrder+1, GridPartType::dimension >::power;
      typedef AgglomerateDGMapper< GridPartType > BlockMapperType;

      template< class DiscreteFunction, class Operation = DFCommunicationOperation::Copy >
      struct CommDataHandle
      {
        typedef Operation OperationType;
        typedef DefaultCommunicationHandler< DiscreteFunction, Operation > Type:
      };
    };



    // AgglomerateDGSpace
    // ------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    class AgglomerateDGSpace
      : public DiscreteFunctionSpaceDefault< AgglomerateDGSpaceTraits< FunctionSpace, GridPart, polOrder > >
    {
      typedef AgglomerateDGSpace< FunctionSpace, GridPart, polOrder > ThisType;
      typedef DiscreteFunctionSpaceDefault< AgglomerateDGSpaceTraits< FunctionSpace, GridPart, polOrder > > BaseType;

    public:
      typedef Agglomeration< GridPart > AgglomerationType;

      typedef typename BaseType::BasisFunctionSetType BasisFunctionSetType;

      typedef typename BaseType::BlockMapperType BlockMapperType;

      typedef typename BaseType::EntityType EntityType;
      typedef typename BaseType::GridPartType GridPartType;

      AgglomerateDGSpace ( GridPartType &gridPart, const AgglomerationType &agglomeration )
        : BaseType( gridPart ),
          blockMapper_( agglomeration ),
          boundingBoxes_( boundingBoxes( agglomeration ) ),
          scalarShapeFunctionSet_( polOrder )
      {}

      const BasisFunctionSetType basisFunctionSet ( const EntityType &entity ) const
      {
        typename Traits::ShapeFunctionSetType shapeFunctionSet( &scalarShapeFunctionSet );
        return BasisFunctionSetType( entity, boundingBoxes_[ agglomeration().index( entity ) ], std::move( shapeFunctionSet ) );
      }

      BlockMapperType &blockMapper () const { return blockMapper_; }

      // extra interface methods

      static constexpr bool continuous () noexcept { return false; }

      static constexpr bool continuous ( const typename BaseType::IntersectionType & ) noexcept { return false; }

      static constexpr int order ( const EntityType & ) noexcept { return polOrder; }
      static constexpr int order () { return polOrder; }

      static constexpr Fem::DFSpaceIdentifier type () noexcept { return GenericSpace_id; }

      // implementation-defined methods

      const AgglomerationType &agglomeration () const { return blockMapper_.agglomeration(); }

    private:
      mutable BlockMapperType blockMapper_;
      std::vector< BoundingBox< GridPart > > boundingBoxes_;
      typename Traits::ScalarShapeFunctionSetType scalarShapeFunctionSet_;
    };

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_DGSPACE_HH
