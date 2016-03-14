#ifndef DUNE_VEM_SPACE_AGGLOMERATESPACE_HH
#define DUNE_VEM_SPACE_AGGLOMERATESPACE_HH

#include <utility>

#include <dune/common/power.hh>

#include <dune/fem/space/common/commoperations.hh>
#include <dune/fem/space/common/defaultcommhandler.hh>
#include <dune/fem/space/common/discretefunctionspace.hh>
#include <dune/fem/space/common/functionspace.hh>
#include <dune/fem/space/shapefunctionset/orthonormal.hh>
#include <dune/fem/space/shapefunctionset/proxy.hh>
#include <dune/fem/space/shapefunctionset/vectorial.hh>

#include <dune/vem/agglomeration/boundingbox.hh>
#include <dune/vem/agglomeration/dgmapper.hh>
#include <dune/vem/space/basisfunctionset.hh>

namespace Dune
{

  namespace Vem
  {

    // Internal Forward Declarations
    // -----------------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    class AgglomerationVEMSpace;



    // AgglomerationVEMSpaceTraits
    // ---------------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    struct AgglomerationVEMSpaceTraits
    {
      friend class AgglomerationVEMSpace< FunctionSpace, GridPart, polOrder >;

      typedef AgglomerationVEMSpace< FunctionSpace, GridPart, polOrder > DiscreteFunctionSpaceType;

      typedef FunctionSpace FunctionSpaceType;
      typedef GridPart GridPartType;

      static const int codimension = 0;

    private:
      typedef typename GridPartType::template Codim< codimension >::EntityType EntityType;

      typedef typename Fem::FunctionSpace< typename FunctionSpaceType::DomainFieldType, typename FunctionSpaceType::RangeFieldType, FunctionSpaceType::dimDomain, 1 > ScalarFunctionSpaceType;
      typedef Fem::OrthonormalShapeFunctionSet< ScalarFunctionSpaceType, polOrder > ScalarShapeFunctionSetType;
      typedef Fem::VectorialShapeFunctionSet< Fem::ShapeFunctionSetProxy< ScalarShapeFunctionSetType >, typename FunctionSpaceType::RangeType > ShapeFunctionSetType;

    public:
      typedef VEMBasisFunctionSet< EntityType, ShapeFunctionSetType > BasisFunctionSetType;

      static const std::size_t localBlockSize = FunctionSpaceType::dimRange * StaticPower< polOrder+1, GridPartType::dimension >::power;
      typedef AgglomerationDGMapper< GridPartType > BlockMapperType;

      template< class DiscreteFunction, class Operation = Fem::DFCommunicationOperation::Copy >
      struct CommDataHandle
      {
        typedef Operation OperationType;
        typedef Fem::DefaultCommunicationHandler< DiscreteFunction, Operation > Type;
      };
    };



    // AgglomerationVEMSpace
    // ---------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    class AgglomerationVEMSpace
      : public Fem::DiscreteFunctionSpaceDefault< AgglomerationVEMSpaceTraits< FunctionSpace, GridPart, polOrder > >
    {
      typedef AgglomerationVEMSpace< FunctionSpace, GridPart, polOrder > ThisType;
      typedef Fem::DiscreteFunctionSpaceDefault< AgglomerationVEMSpaceTraits< FunctionSpace, GridPart, polOrder > > BaseType;

    public:
      typedef typename BaseType::Traits Traits;

      typedef Agglomeration< GridPart > AgglomerationType;

      typedef typename BaseType::BasisFunctionSetType BasisFunctionSetType;

      typedef typename BaseType::BlockMapperType BlockMapperType;

      typedef typename BaseType::EntityType EntityType;
      typedef typename BaseType::GridPartType GridPartType;

      AgglomerationVEMSpace ( GridPartType &gridPart, const AgglomerationType &agglomeration )
        : BaseType( gridPart ),
          blockMapper_( agglomeration ),
          boundingBoxes_( boundingBoxes( agglomeration ) ),
          scalarShapeFunctionSet_( Dune::GeometryType( Dune::GeometryType::cube, GridPart::dimension ) )
      {
        buildProjections();
      }

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
      void buildProjections ();

      mutable BlockMapperType blockMapper_;
      std::vector< BoundingBox< GridPart > > boundingBoxes_;
      std::vector< ValueProjection > valueProjections_;
      std::vector< JacobianProjection > jacobianProjections_;
      typename Traits::ScalarShapeFunctionSetType scalarShapeFunctionSet_;
    };



    template< class FunctionSpace, class GridPart, int polOrder >
    inline void AgglomerationVEMSpace< FunctionSpace, GridPart, polOrder >::buildProjections ()
    {
      typedef typename GridPart::Codim< 0 >::EntitySeedType EntitySeed;
      std::vector< std::vector< EntitySeed > > entitySeeds( agglomeration().size() );
      for( const auto &element : elements( static_cast< typename GridPart::GridViewType >( gridPart ), Dune::Partitions::interiorBorder ) )
        entitySeed[ agglomeration().index( element ) ].push_back( element.seed() );

      const std::size_t numShapeFunctions = scalarShapeFunctionSet_.size();
      ValueProjection DT( numShapeFunctions );
      DynamicMatrix< DomainFieldType > DTD( numShapeFunctions, numShapeFunctions );

      for( std::size_t agglomerate = 0; agglomerate < agglomeration.size(); ++agglomerate )
      {
        const std::size_t numSubAgglomerates = agIndexSet.subAgglomerates( element, GridPart::dimension );
        for( auto &d : DT )
          d.resize( numSubAgglomerates );

        for( const EntitySeed &entitySeed : entitySeeds[ agglomerate ] )
        {
          const auto &element = gridPart.entity( entitySeed );

          const auto &refElement = ReferenceElements< typename GridPart::ctype, GridPart::dimension >::general( element.type() );

          for( int i = 0; i < refElement.size( GridPart::dimension ); ++i )
          {
            const int k = agIndexSet.localIndex( element, i, GridPart::dimension );
            if( k == -1 )
              continue;

            DomainType x = element.geometry().corner( i ) - bbox_.first;
            for( int k = 0; k < GridPartType::dimensionworld; ++k )
              x[ k ] /= (bbox_.second[ k ] - bbox_.first[ k ]);

            scalarShapeFunctionSet.evaluateEach( x, [ &DT, k ] ( std::size_t j, FieldVector< DomainFieldType, 1 > phi ) {
                DT[ j ][ k ] = phi[ 0 ];
              } );
          }

          for( const auto &intersection : intersections( static_cast< typename GridPart::GridViewType >( gridPart ), element ) )
          {
            if( !intersection.boundary() && (agglomeration().index( Dune::Fem::make_entity( intersection.outside() ) ) == agglomerate) )
              continue;

            const int numVertices = refElement.size( intersection.indexInInside(), 1, GridPart::dimension );
            for( int i = 0; i < numVertices; ++i )
            {
              const int j = refElement.subEntity( intersection.indexInInside(), 1, i, GridPart::dimension );
              const int k = agIndexSet.localIndex( element, j, GridPart::dimension );
              Pi0X[ k ].axpy( 0.5*intersection.geometry().volume(), intersection.centerUnitOuterNormal() );
            }
          }
        }

        for( std::size_t i = 0; i < numShapeFunctions; ++i )
          for( std::size_t j = 0; j < numShapeFunctions; ++j )
          {
            DTD[ i ][ j ] = 0;
            for( std::size_t k = 0; k < numSubAgglomerates; ++k )
              DTD[ i ][ j ] += DT[ i ][ k ] * DT[ j ][ k ];
          }
        DTD.invert();

        valueProjections_[ agglomerate ].resize( numShapeFunctions );
        for( std::size_t i = 0; i < numShapeFunctions; ++i )
        {
          valueProjections_[ i ].resize( numSubAgglomerates, 0 );
          for( std::size_t k = 0; k < numShapeFunctions; ++k )
            for( std::size_t j = 0; j < numSubAgglomerates; ++j )
              valueProjections_[ i ][ j ] += DTD[ i ][ k ] * DT[ k ][ j ];
        }
      }
    }

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_AGGLOMERATESPACE_HH
