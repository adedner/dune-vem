#ifndef DUNE_VEM_SPACE_AGGLOMERATION_HH
#define DUNE_VEM_SPACE_AGGLOMERATION_HH

#include <cassert>

#include <utility>

#include <dune/fem/quadrature/elementquadrature.hh>
#include <dune/fem/space/common/commoperations.hh>
#include <dune/fem/space/common/defaultcommhandler.hh>
#include <dune/fem/space/common/discretefunctionspace.hh>
#include <dune/fem/space/common/functionspace.hh>
#include <dune/fem/space/shapefunctionset/orthonormal.hh>
#include <dune/fem/space/shapefunctionset/proxy.hh>
#include <dune/fem/space/shapefunctionset/vectorial.hh>

#include <dune/vem/agglomeration/boundingbox.hh>
#include <dune/vem/agglomeration/dofmapper.hh>
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

      static const std::size_t localBlockSize = FunctionSpaceType::dimRange;
      typedef AgglomerationDofMapper< GridPartType > BlockMapperType;

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
      typedef AgglomerationIndexSet< GridPart > AgglomerationIndexSetType;

      typedef typename BaseType::BasisFunctionSetType BasisFunctionSetType;

      typedef typename BaseType::BlockMapperType BlockMapperType;

      typedef typename BaseType::EntityType EntityType;
      typedef typename BaseType::GridPartType GridPartType;

      using BaseType::gridPart;

      explicit AgglomerationVEMSpace ( GridPartType &gridPart, const AgglomerationIndexSetType &agIndexSet )
        : BaseType( gridPart ),
          blockMapper_( agIndexSet, { std::make_pair( GridPart::dimension, 1u ) } ),
          boundingBoxes_( boundingBoxes( agIndexSet.agglomeration() ) ),
          scalarShapeFunctionSet_( Dune::GeometryType( Dune::GeometryType::cube, GridPart::dimension ) )
      {
        buildProjections();
      }

      const BasisFunctionSetType basisFunctionSet ( const EntityType &entity ) const
      {
        typename Traits::ShapeFunctionSetType shapeFunctionSet( &scalarShapeFunctionSet_ );
        const std::size_t agglomerate = agglomeration().index( entity );
        const auto &bbox = boundingBoxes_[ agglomerate ];
        const auto &valueProjection = valueProjections_[ agglomerate ];
        const auto &jacobianProjection = jacobianProjections_[ agglomerate ];
        return BasisFunctionSetType( entity, bbox, valueProjection, jacobianProjection, std::move( shapeFunctionSet ) );
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
      std::vector< typename BasisFunctionSetType::ValueProjection > valueProjections_;
      std::vector< typename BasisFunctionSetType::JacobianProjection > jacobianProjections_;
      typename Traits::ScalarShapeFunctionSetType scalarShapeFunctionSet_;
    };



    template< class FunctionSpace, class GridPart, int polOrder >
    inline void AgglomerationVEMSpace< FunctionSpace, GridPart, polOrder >::buildProjections ()
    {
      typedef typename BasisFunctionSetType::DomainFieldType DomainFieldType;
      typedef typename BasisFunctionSetType::DomainType DomainType;
      typedef typename GridPart::template Codim< 0 >::EntitySeedType EntitySeed;

      std::vector< std::vector< EntitySeed > > entitySeeds( agglomeration().size() );
      for( const auto &element : elements( static_cast< typename GridPart::GridViewType >( gridPart() ), Partitions::interiorBorder ) )
        entitySeeds[ agglomeration().index( element ) ].push_back( element.seed() );

      const std::size_t numShapeFunctions = scalarShapeFunctionSet_.size();
      typename BasisFunctionSetType::ValueProjection DT( numShapeFunctions );
      DynamicMatrix< DomainFieldType > DTD( numShapeFunctions, numShapeFunctions );
      std::vector< DomainType > pi0XT;

      valueProjections_.resize( agglomeration().size() );
      jacobianProjections_.resize( agglomeration().size() );

      for( std::size_t agglomerate = 0; agglomerate < agglomeration().size(); ++agglomerate )
      {
        const auto &bbox = boundingBoxes_[ agglomerate ];

        const std::size_t numSubAgglomerates = blockMapper().indexSet().subAgglomerates( agglomerate, GridPart::dimension );
        for( auto &row : DT )
          row.resize( numSubAgglomerates );
        pi0XT.resize( numSubAgglomerates );

        DomainFieldType H0 = 0;
        for( const EntitySeed &entitySeed : entitySeeds[ agglomerate ] )
        {
          const auto &element = gridPart().entity( entitySeed );
          const auto geometry = element.geometry();

          Fem::ElementQuadrature< GridPart, 0 > quadrature( element, 0 );
          for( std::size_t qp = 0; qp < quadrature.nop(); ++qp )
          {
            DomainType x = geometry.global( quadrature.point( qp ) ) - bbox.first;
            for( int k = 0; k < GridPartType::dimensionworld; ++k )
              x[ k ] /= (bbox.second[ k ] - bbox.first[ k ]);

            const DomainFieldType weight = geometry.integrationElement( quadrature.point( qp ) ) * quadrature.weight( qp );
            scalarShapeFunctionSet_.evaluateEach( x, [ &H0, weight ] ( std::size_t alpha, FieldVector< DomainFieldType, 1 > phi ) {
                if( alpha == 0 )
                  H0 += weight * phi[ 0 ];
              } );
          }

          const auto &refElement = ReferenceElements< typename GridPart::ctype, GridPart::dimension >::general( element.type() );

          for( int i = 0; i < refElement.size( GridPart::dimension ); ++i )
          {
            const int k = blockMapper().indexSet().localIndex( element, i, GridPart::dimension );
            if( k == -1 )
              continue;

            DomainType x = geometry.corner( i ) - bbox.first;
            for( int k = 0; k < GridPartType::dimensionworld; ++k )
              x[ k ] /= (bbox.second[ k ] - bbox.first[ k ]);

            scalarShapeFunctionSet_.evaluateEach( x, [ &DT, k ] ( std::size_t alpha, FieldVector< DomainFieldType, 1 > phi ) {
                DT[ alpha ][ k ] = phi[ 0 ];
              } );
          }

          for( const auto &intersection : intersections( static_cast< typename GridPart::GridViewType >( gridPart() ), element ) )
          {
            if( !intersection.boundary() && (agglomeration().index( Dune::Fem::make_entity( intersection.outside() ) ) == agglomerate) )
              continue;
            assert( intersection.conforming() );

            const int faceIndex = intersection.indexInInside();
            const int numEdgeVertices = refElement.size( faceIndex, 1, GridPart::dimension );
            const DomainFieldType iVolume = intersection.geometry().volume();
            const DomainType outerNormal = intersection.centerUnitOuterNormal();
            for( int i = 0; i < numEdgeVertices; ++i )
            {
              const int j = refElement.subEntity( faceIndex, 1, i, GridPart::dimension );
              const int k = blockMapper().indexSet().localIndex( element, j, GridPart::dimension );
              pi0XT[ k ].axpy( 0.5*iVolume, outerNormal );
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

        auto &valueProjection = valueProjections_[ agglomerate ];
        valueProjection.resize( numShapeFunctions );
        for( std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha )
        {
          valueProjection[ alpha ].resize( numSubAgglomerates, 0 );
          for( std::size_t beta = 0; beta < numShapeFunctions; ++beta )
            for( std::size_t j = 0; j < numSubAgglomerates; ++j )
              valueProjection[ alpha ][ j ] += DTD[ alpha ][ beta ] * DT[ beta ][ j ];
        }

        // Warning: This is a dirty hack
        auto &jacobianProjection = jacobianProjections_[ agglomerate ];
        jacobianProjection.resize( numShapeFunctions );
        jacobianProjection[ 0 ] = pi0XT;
        std::transform( jacobianProjection[ 0 ].begin(), jacobianProjection[ 0 ].end(), jacobianProjection[ 0 ].begin(),
                        [ H0 ] ( DomainType x ) { return x *= (1 / H0); } );
        for( std::size_t alpha = 1; alpha < numShapeFunctions; ++alpha )
          jacobianProjection[ alpha ].resize( numSubAgglomerates, DomainType( 0 ) );
      }
    }

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_AGGLOMERATION_HH
