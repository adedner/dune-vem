#ifndef DUNE_VEM_AGGLOMERATION_DGSPACE_HH
#define DUNE_VEM_AGGLOMERATION_DGSPACE_HH

#include <utility>

#include <dune/common/power.hh>
#include <dune/common/version.hh>

#if DUNE_VERSION_NEWER(DUNE_FEM, 2, 6)
#include <dune/fem/common/hybrid.hh>
#endif // #if DUNE_VERSION_NEWER(DUNE_FEM, 2, 6)

#include <dune/fem/space/common/commoperations.hh>
#include <dune/fem/space/common/defaultcommhandler.hh>
#include <dune/fem/space/common/discretefunctionspace.hh>
#include <dune/fem/space/common/functionspace.hh>
// #include <dune/fem/space/shapefunctionset/legendre.hh>
#include <dune/fem/space/shapefunctionset/orthonormal.hh>
#include <dune/fem/space/shapefunctionset/proxy.hh>
#include <dune/fem/space/shapefunctionset/vectorial.hh>
#include <dune/fem/solver/cginverseoperator.hh>
#include <dune/fem/operator/linear/spoperator.hh>

#include <dune/vem/agglomeration/basisfunctionset.hh>
#include <dune/vem/agglomeration/boundingbox.hh>
#include <dune/vem/agglomeration/dgmapper.hh>
#include <dune/vem/operator/mass.hh>

namespace Dune
{

  namespace Vem
  {

    // Internal Forward Declarations
    // -----------------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    class AgglomerationDGSpace;

    // IsAgglomerationDGSpace
    // ------------------------------

    template< class DiscreteFunctionSpace >
    struct IsAgglomerationDGSpace
      : std::integral_constant< bool, false >
    {};

    template< class FunctionSpace, class GridPart, int order >
    struct IsAgglomerationDGSpace< AgglomerationDGSpace< FunctionSpace, GridPart, order > >
      : std::integral_constant< bool, true >
    {};


    // AgglomerationDGSpaceTraits
    // --------------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    struct AgglomerationDGSpaceTraits
    {
      friend class AgglomerationDGSpace< FunctionSpace, GridPart, polOrder >;

      typedef AgglomerationDGSpace< FunctionSpace, GridPart, polOrder > DiscreteFunctionSpaceType;

      typedef FunctionSpace FunctionSpaceType;
      typedef GridPart GridPartType;

      static const int codimension = 0;

    private:
      typedef typename GridPartType::template Codim< codimension >::EntityType EntityType;

      /*
      typedef typename Fem::FunctionSpace< typename FunctionSpaceType::DomainFieldType, typename FunctionSpaceType::RangeFieldType, FunctionSpaceType::dimDomain, 1 > ScalarFunctionSpaceType;
      // typedef Fem::LegendreShapeFunctionSet< ScalarFunctionSpaceType > ScalarShapeFunctionSetType;
      typedef Fem::OrthonormalShapeFunctionSet< ScalarFunctionSpaceType > ScalarShapeFunctionSetType;
      */

    public:
      typedef Dune::Fem::FunctionSpace<
          typename FunctionSpace::DomainFieldType, typename FunctionSpace::RangeFieldType,
           GridPartType::dimension, 1
        > ScalarShapeFunctionSpaceType;

      struct ScalarShapeFunctionSet
        : public Dune::Fem::OrthonormalShapeFunctionSet< ScalarShapeFunctionSpaceType >
      {
        typedef Dune::Fem::OrthonormalShapeFunctionSet< ScalarShapeFunctionSpaceType >   BaseType;

        static constexpr int numberShapeFunctions =
              Dune::Fem::OrthonormalShapeFunctions< ScalarShapeFunctionSpaceType::dimDomain >::size(polOrder);
      public:
        explicit ScalarShapeFunctionSet ( Dune::GeometryType type )
          : BaseType( type, polOrder )
        {
          assert( size() == BaseType::size() );
        }
        explicit ScalarShapeFunctionSet ( Dune::GeometryType type, int p )
          : BaseType( type, p )
        {
          assert( size() == BaseType::size() );
        }

        // overload size method because it's a static value
        static constexpr unsigned int size() { return numberShapeFunctions; }
      };
      typedef ScalarShapeFunctionSet ScalarShapeFunctionSetType;
      typedef Fem::VectorialShapeFunctionSet< Fem::ShapeFunctionSetProxy< ScalarShapeFunctionSetType >, typename FunctionSpaceType::RangeType > ShapeFunctionSetType;

      typedef BoundingBoxBasisFunctionSet< EntityType, ShapeFunctionSetType > BasisFunctionSetType;

      // static const std::size_t localBlockSize = FunctionSpaceType::dimRange * StaticPower< polOrder+1, GridPartType::dimension >::power;
      typedef Hybrid::IndexRange< int, FunctionSpaceType::dimRange * ScalarShapeFunctionSet::numberShapeFunctions > LocalBlockIndices;
      typedef AgglomerationDGMapper< GridPartType > BlockMapperType;

      template< class DiscreteFunction, class Operation = Fem::DFCommunicationOperation::Copy >
      struct CommDataHandle
      {
        typedef Operation OperationType;
        typedef Fem::DefaultCommunicationHandler< DiscreteFunction, Operation > Type;
      };
    };



    // AgglomerationDGSpace
    // --------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    class AgglomerationDGSpace
      : public Fem::DiscreteFunctionSpaceDefault< AgglomerationDGSpaceTraits< FunctionSpace, GridPart, polOrder > >
    {
      typedef AgglomerationDGSpace< FunctionSpace, GridPart, polOrder > ThisType;
      typedef Fem::DiscreteFunctionSpaceDefault< AgglomerationDGSpaceTraits< FunctionSpace, GridPart, polOrder > > BaseType;

    public:
      typedef typename BaseType::Traits Traits;

      typedef Agglomeration< GridPart > AgglomerationType;

      typedef typename BaseType::BasisFunctionSetType BasisFunctionSetType;

      typedef typename BaseType::BlockMapperType BlockMapperType;

      typedef typename BaseType::EntityType EntityType;
      typedef typename BaseType::GridPartType GridPartType;

      enum { hasLocalInterpolate = false };

      AgglomerationDGSpace ( const AgglomerationType &agglomeration )
        : BaseType( agglomeration.gridPart() ),
          blockMapper_( agglomeration ),
          boundingBoxes_( boundingBoxes( agglomeration ) ),
          // scalarShapeFunctionSet_( polOrder )
          scalarShapeFunctionSet_(
              Dune::GeometryType(Dune::GeometryType::cube,GridPart::dimension), polOrder )
      {}

      const BasisFunctionSetType basisFunctionSet ( const EntityType &entity ) const
      {
        typename Traits::ShapeFunctionSetType shapeFunctionSet( &scalarShapeFunctionSet_ );
        return BasisFunctionSetType( entity, boundingBoxes_[ agglomeration().index( entity ) ], std::move( shapeFunctionSet ) );
      }

      BlockMapperType &blockMapper () const { return blockMapper_; }

      // extra interface methods

      static constexpr bool continuous () noexcept { return false; }

      static constexpr bool continuous ( const typename BaseType::IntersectionType & ) noexcept { return false; }

      static constexpr int order ( const EntityType & ) noexcept { return polOrder; }
      static constexpr int order () { return polOrder; }

      static constexpr Fem::DFSpaceIdentifier type () noexcept { return Fem::GenericSpace_id; }

      // implementation-defined methods

      const AgglomerationType &agglomeration () const { return blockMapper_.agglomeration(); }

    private:
      mutable BlockMapperType blockMapper_;
      std::vector< BoundingBox< GridPart > > boundingBoxes_;
      typename Traits::ScalarShapeFunctionSetType scalarShapeFunctionSet_;
    };

  } // namespace Vem

  namespace Fem
  {
    template< class GridFunction, class DiscreteFunction, unsigned int partitions >
    static inline std::enable_if_t< std::is_convertible< GridFunction, HasLocalFunction >::value && Dune::Vem::IsAgglomerationDGSpace< typename DiscreteFunction::DiscreteFunctionSpaceType >::value >
    interpolate ( const GridFunction &u, DiscreteFunction &v, PartitionSet< partitions > ps )
    {
      // !!! a very crude implementation - should be done locally on each polygon
      v.clear();
      DiscreteFunction rhs( v );
      rhs.clear();
      Dune::Vem::applyMass( u, rhs );
      typedef Dune::Fem::SparseRowLinearOperator< DiscreteFunction, DiscreteFunction > LinearOperator;
      LinearOperator assembledMassOp( "assembled mass operator", v.space(), v.space() );
      Dune::Vem::MassOperator< LinearOperator > massOp( v.space() );
      massOp.jacobian( v, assembledMassOp );
      Dune::Fem::CGInverseOperator< DiscreteFunction > invOp( assembledMassOp, 1e-8, 1e-8, true );
      invOp( rhs, v );
    }
  } // namespace Fem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_DGSPACE_HH
