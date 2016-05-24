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
#include <dune/vem/agglomeration/shapefunctionset.hh>
#include <dune/vem/misc/compatibility.hh>
#include <dune/vem/space/basisfunctionset.hh>
#include <dune/vem/space/interpolation.hh>

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

    private:
      typedef AgglomerationVEMInterpolation< AgglomerationIndexSetType > AgglomerationInterpolationType;
      typedef typename Traits::ScalarShapeFunctionSetType ScalarShapeFunctionSetType;

    public:
      typedef typename BaseType::BasisFunctionSetType BasisFunctionSetType;

      typedef typename BaseType::BlockMapperType BlockMapperType;

      typedef typename BaseType::EntityType EntityType;
      typedef typename BaseType::GridPartType GridPartType;

      typedef DynamicMatrix< typename BasisFunctionSetType::DomainFieldType > Stabilization;

      using BaseType::gridPart;

      explicit AgglomerationVEMSpace ( GridPartType &gridPart, const AgglomerationIndexSetType &agIndexSet )
        : BaseType( gridPart ),
          blockMapper_( agIndexSet, AgglomerationInterpolationType::dofsPerCodim() ),
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

      const Stabilization &stabilization ( const EntityType &entity ) const { return stabilizations_[ agglomeration().index( entity ) ]; }

    private:
      void buildProjections ();

      mutable BlockMapperType blockMapper_;
      std::vector< BoundingBox< GridPart > > boundingBoxes_;
      std::vector< typename BasisFunctionSetType::ValueProjection > valueProjections_;
      std::vector< typename BasisFunctionSetType::JacobianProjection > jacobianProjections_;
      std::vector< Stabilization > stabilizations_;
      ScalarShapeFunctionSetType scalarShapeFunctionSet_;
    };



    // Implementation of AgglomerationVEMSpace
    // ---------------------------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    inline void AgglomerationVEMSpace< FunctionSpace, GridPart, polOrder >::buildProjections ()
    {
      typedef typename BasisFunctionSetType::DomainFieldType DomainFieldType;
      typedef typename BasisFunctionSetType::DomainType DomainType;
      typedef typename GridPart::template Codim< 0 >::EntityType ElementType;
      typedef typename GridPart::template Codim< 0 >::EntitySeedType ElementSeedType;

      std::vector< std::vector< ElementSeedType > > entitySeeds( agglomeration().size() );
      for( const ElementType &element : elements( static_cast< typename GridPart::GridViewType >( gridPart() ), Partitions::interiorBorder ) )
        entitySeeds[ agglomeration().index( element ) ].push_back( element.seed() );

      const std::size_t numShapeFunctions = scalarShapeFunctionSet_.size();
      DynamicMatrix< DomainFieldType > D;
      DynamicMatrix< DomainFieldType > DTD( numShapeFunctions, numShapeFunctions );
      std::vector< DomainType > pi0XT;

      valueProjections_.resize( agglomeration().size() );
      jacobianProjections_.resize( agglomeration().size() );
      stabilizations_.resize( agglomeration().size() );

      AgglomerationInterpolationType interpolation( blockMapper().indexSet() );
      for( std::size_t agglomerate = 0; agglomerate < agglomeration().size(); ++agglomerate )
      {
        const auto &bbox = boundingBoxes_[ agglomerate ];

        const std::size_t numDofs = blockMapper().numDofs( agglomerate );
        D.resize( numDofs, numShapeFunctions );
        pi0XT.resize( numDofs );

        DomainFieldType H0 = 0;
        std::fill( pi0XT.begin(), pi0XT.end(), DomainType( 0 ) );
        for( const ElementSeedType &entitySeed : entitySeeds[ agglomerate ] )
        {
          const ElementType &element = gridPart().entity( entitySeed );
          const auto geometry = element.geometry();

          BoundingBoxShapeFunctionSet< ElementType, ScalarShapeFunctionSetType > shapeFunctionSet( element, bbox, scalarShapeFunctionSet_ );

          Fem::ElementQuadrature< GridPart, 0 > quadrature( element, 0 );
          for( std::size_t qp = 0; qp < quadrature.nop(); ++qp )
          {
            const DomainFieldType weight = geometry.integrationElement( quadrature.point( qp ) ) * quadrature.weight( qp );
            shapeFunctionSet.evaluateEach( quadrature[ qp ], [ &H0, weight ] ( std::size_t alpha, FieldVector< DomainFieldType, 1 > phi ) {
                if( alpha == 0 )
                  H0 += weight * phi[ 0 ];
              } );
          }

          interpolation( shapeFunctionSet, D );

#if 0
          if( polOrder > 1 )
          {
            const auto &idSet = agglomeration().gridPart().grid().globalIdSet();
            for( int i = 0; i < refElement.size( GridPart::dimension-1 ); ++i )
            {
              const int k = blockMapper().indexSet().localIndex( element, i, 1 );
              if( k == -1 )
                continue;

              const auto left = idSet.subId( element, refElement.subEntity( i, dimension-1, 0, dimension ), dimension );
              const auto right = idSet.subId( element, refElement.subEntity( i, dimension-1, 1, dimension ), dimension );

              const auto subEntity = element.template subEntity< GridPart::dimension-1 >( i );
              const auto geometry = subEntity.geometry();

              for( std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha )
                DT[ alpha ][ k ] = 0;

              typedef Dune::QuadratureRules< typename GridPart::ctype, 1 > QuadratureRules;
              for( const auto &qp : QuadratureRules::rule( refElement.type( i, GridPart::dimension-1 ) ), order )
              {
                typedef typename decltype( geometry )::LocalCoordinate LocalCoordinate;
                LocalCoordinate z = (left < right ? qp.position() : LocalCoordinate{ 1 } - qp.position());
                DomainType x = geometry.global( z ) - bbox.first;
                for( int k = 0; k < GridPartType::dimensionworld; ++k )
                  x[ k ] /= (bbox.second[ k ] - bbox.first[ k ]);

                DomainFieldType weight = qp.weight() * geometry.integrationElement( z );
                scalarShapeFunctionSet_.evaluateEach( x, [ &DT, k, weight ] ( std::size_t alpha, FieldVector< DomainFieldType, 1 > phi ) {
                    DT[ alpha ][ k ] += weight * phi[ 0 ];
                  } );
              }

              for( std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha )
                DT[ alpha ][ k ] *= DomainFieldType( 1 ) / geometry.volume();
            }
          }
  #endif

          const auto &refElement = ReferenceElements< typename GridPart::ctype, GridPart::dimension >::general( element.type() );
          for( const auto &intersection : intersections( static_cast< typename GridPart::GridViewType >( gridPart() ), element ) )
          {
            if( !intersection.boundary() && (agglomeration().index( make_entity( intersection.outside() ) ) == agglomerate) )
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
              assert( k >= 0 );
              pi0XT[ k ].axpy( 0.5*iVolume, outerNormal );
            }
          }
        }

        for( std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha )
        {
          for( std::size_t beta = 0; beta < numShapeFunctions; ++beta )
            DTD[ alpha ][ beta ] = 0;
          for( std::size_t k = 0; k < numDofs; ++k )
            for( std::size_t beta = 0; beta < numShapeFunctions; ++beta )
              DTD[ alpha ][ beta ] += D[ k ][ alpha ] * D[ k ][ beta ];
        }
        DTD.invert();

        auto &valueProjection = valueProjections_[ agglomerate ];
        valueProjection.resize( numShapeFunctions );
        for( std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha )
        {
          valueProjection[ alpha ].resize( numDofs, 0 );
          for( std::size_t beta = 0; beta < numShapeFunctions; ++beta )
            for( std::size_t j = 0; j < numDofs; ++j )
              valueProjection[ alpha ][ j ] += DTD[ alpha ][ beta ] * D[ j ][ beta ];
        }

        Stabilization S( numDofs, numDofs, 0 );
        for( std::size_t i = 0; i < numDofs; ++i )
          S[ i ][ i ] = DomainFieldType( 1 );
        for( std::size_t i = 0; i < numDofs; ++i )
          for( std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha )
            for( std::size_t j = 0; j < numDofs; ++j )
              S[ i ][ j ] -= D[ i ][ alpha ] * valueProjection[ alpha ][ j ];
        Stabilization &stabilization = stabilizations_[ agglomerate ];
        stabilization.resize( numDofs, numDofs, 0 );
        for( std::size_t k = 0; k < numDofs; ++k )
          for( std::size_t i = 0; i < numDofs; ++i )
            for( std::size_t j = 0; j < numDofs; ++j )
              stabilization[ i ][ j ] += S[ k ][ i ] * S[ k ][ j ];

        // Warning: This is a dirty hack
        auto &jacobianProjection = jacobianProjections_[ agglomerate ];
        jacobianProjection.resize( numShapeFunctions );
        jacobianProjection[ 0 ] = pi0XT;
        std::transform( jacobianProjection[ 0 ].begin(), jacobianProjection[ 0 ].end(), jacobianProjection[ 0 ].begin(),
                        [ H0 ] ( DomainType x ) { return x *= (1 / H0); } );
        for( std::size_t alpha = 1; alpha < numShapeFunctions; ++alpha )
          jacobianProjection[ alpha ].resize( numDofs, DomainType( 0 ) );
      }
    }

  } // namespace Vem



  namespace Fem
  {

    // External Forward Declarations
    // -----------------------------

#if HAVE_DUNE_ISTL
    template< class Matrix, class Space >
    struct ISTLParallelMatrixAdapter;

    template< class Matrix >
    class LagrangeParallelMatrixAdapter;
#endif // #if HAVE_DUNE_ISTL



    // ISTLParallelMatrixAdapter for AgglomerationVEMSpace
    // ---------------------------------------------------

#if HAVE_DUNE_ISTL
    template< class Matrix, class FunctionSpace, class GridPart, int polOrder >
    struct ISTLParallelMatrixAdapter< Matrix, Vem::AgglomerationVEMSpace< FunctionSpace, GridPart, polOrder > >
    {
      typedef LagrangeParallelMatrixAdapter< Matrix > Type;
    };
#endif // #if HAVE_DUNE_ISTL

  } // namespace Fem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_AGGLOMERATION_HH
