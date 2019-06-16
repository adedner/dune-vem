#ifndef DUNE_VEM_SPACE_AGGLOMERATION_HH
#define DUNE_VEM_SPACE_AGGLOMERATION_HH

#include <cassert>

#include <utility>

#include <dune/fem/common/hybrid.hh>

#include <dune/fem/quadrature/elementquadrature.hh>
#include <dune/fem/space/common/commoperations.hh>
#include <dune/fem/space/common/defaultcommhandler.hh>
#include <dune/fem/space/common/discretefunctionspace.hh>
#include <dune/fem/space/common/functionspace.hh>
#include <dune/fem/space/basisfunctionset/vectorial.hh>
#include <dune/fem/space/shapefunctionset/orthonormal.hh>
#include <dune/fem/space/shapefunctionset/proxy.hh>
#include <dune/fem/space/shapefunctionset/vectorial.hh>
#include <dune/fem/space/common/capabilities.hh>

#include <dune/vem/agglomeration/dofmapper.hh>
// #include <dune/vem/agglomeration/shapefunctionset.hh>
#include <dune/vem/misc/compatibility.hh>
#include <dune/vem/misc/pseudoinverse.hh>
#include <dune/vem/agglomeration/dgspace.hh>
#include <dune/vem/space/basisfunctionset.hh>
#include <dune/vem/space/interpolation.hh>
#include <dune/vem/space/interpolate.hh>

#include <dune/vem/space/test.hh>

namespace Dune
{

  namespace Vem
  {

    // Internal Forward Declarations
    // -----------------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    class AgglomerationVEMSpace;



    // IsAgglomerationVEMSpace
    // -----------------------

    template< class DiscreteFunctionSpace >
    struct IsAgglomerationVEMSpace
      : std::integral_constant< bool, false >
    {};

    template< class FunctionSpace, class GridPart, int order >
    struct IsAgglomerationVEMSpace< AgglomerationVEMSpace< FunctionSpace, GridPart, order > >
      : std::integral_constant< bool, true >
    {};



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

    public:
      typedef Dune::Fem::FunctionSpace<
          typename FunctionSpace::DomainFieldType, typename FunctionSpace::RangeFieldType,
           GridPartType::dimension, 1
        > ScalarShapeFunctionSpaceType;
      typedef AgglomerationDGSpaceTraits< ScalarShapeFunctionSpaceType, GridPart, polOrder > DGTraitsType;
      typedef typename DGTraitsType::ScalarShapeFunctionSetType ScalarShapeFunctionSetType;
      typedef typename DGTraitsType::BasisFunctionSetType ScalarBBBasisFunctionSetType;
      typedef VEMBasisFunctionSet< EntityType, ScalarBBBasisFunctionSetType > ScalarBasisFunctionSetType;
      // typedef Fem::VectorialShapeFunctionSet< Fem::SmartShapeFunctionSetProxy< ScalarBasisFunctionSetType >, typename FunctionSpaceType::RangeType > ShapeFunctionSetType;
      // typedef Dune::Fem::SimpleDefaultBasisFunctionSet< EntityType, ShapeFunctionSetType > BasisFunctionSetType;
      typedef Fem::VectorialBasisFunctionSet< ScalarBasisFunctionSetType, typename FunctionSpaceType::RangeType > BasisFunctionSetType;

      typedef Hybrid::IndexRange< int, FunctionSpaceType::dimRange > LocalBlockIndices;
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
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld-1,1> EdgeFSType;
      typedef Dune::Fem::OrthonormalShapeFunctionSet<EdgeFSType> EdgeShapeFunctionSetType;

      typedef DynamicMatrix< typename BasisFunctionSetType::DomainFieldType > Stabilization;

      using BaseType::gridPart;

      enum { hasLocalInterpolate = false };
      static const int polynomialOrder = polOrder;

      // for interpolation
      struct InterpolationType
      {
        InterpolationType( const AgglomerationIndexSetType &indexSet, const EntityType &element ) noexcept
        : inter_(indexSet), element_(element) {}
        template <class U,class V>
        void operator()(const U& u, V& v)
        { inter_(element_, u,v); }
        AgglomerationInterpolationType inter_;
        const EntityType &element_;
      };

      AgglomerationVEMSpace ( AgglomerationType &agglomeration, bool conforming )
        : BaseType( agglomeration.gridPart() ),
          agIndexSet_( agglomeration ),
          blockMapper_( agIndexSet_, AgglomerationInterpolationType::dofsPerCodim(polOrder,conforming) ),
          scalarShapeFunctionSet_( Dune::GeometryType( Dune::GeometryType::cube, GridPart::dimension ) ),
          edgeShapeFunctionSet_(   Dune::GeometryType( Dune::GeometryType::cube, GridPart::dimension-1 ),
              AgglomerationInterpolationType::testSpaces(polOrder,conforming)[1] +       // edge order
                (AgglomerationInterpolationType::testSpaces(polOrder,conforming)[0]+1)*2 // vertex order * number of vertices on edge
              ),
          polOrder_( polOrder ),
          conforming_( conforming )
      {
        buildProjections();
      }

      const bool conforming() const { return conforming_; }
      // const unsigned int polynomialOrder() const { return polOrder_; }

      const BasisFunctionSetType basisFunctionSet ( const EntityType &entity ) const
      {
        const std::size_t agglomerate = agglomeration().index( entity );
        const auto &valueProjection = valueProjections_[ agglomerate ];
        const auto &jacobianProjection = jacobianProjections_[ agglomerate ];
        const auto &bbox = blockMapper_.indexSet().boundingBox(agglomerate);
        // scalar ONB Basis proxy
        typename Traits::DGTraitsType::ShapeFunctionSetType scalarShapeFunctionSet( &scalarShapeFunctionSet_ );
        // scalar BB Basis
        typename Traits::ScalarBBBasisFunctionSetType bbScalarBasisFunctionSet( entity, bbox, std::move( scalarShapeFunctionSet ) );
        // vectorial extended VEM Basis
        /*
        typename Traits::ShapeFunctionSetType shapeFunctionSet( entity, bbox, valueProjection, jacobianProjection, std::move( bbScalarBasisFunctionSet ) );
        return BasisFunctionSetType( entity, std::move( shapeFunctionSet ) );
        */
        typename Traits::ScalarBasisFunctionSetType scalarBFS( entity, bbox, valueProjection, jacobianProjection, std::move( bbScalarBasisFunctionSet ) );
        return BasisFunctionSetType( std::move( scalarBFS ) );
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

      //////////////////////////////////////////////////////////
      // Non-interface methods (used in DirichletConstraints) //
      //////////////////////////////////////////////////////////
      /** \brief return local interpolation for given entity
       *
       *  \param[in]  entity  grid part entity
       */
      InterpolationType interpolation ( const EntityType &entity ) const
      {
        return InterpolationType( blockMapper().indexSet(), entity );
      }

    private:
      void buildProjections ();

      // issue with making these const: use of delete default constructor in some python bindings...
      AgglomerationIndexSetType agIndexSet_;
      mutable BlockMapperType blockMapper_;
      std::vector< typename Traits::ScalarBasisFunctionSetType::ValueProjection > valueProjections_;
      std::vector< typename Traits::ScalarBasisFunctionSetType::JacobianProjection > jacobianProjections_;
      std::vector< Stabilization > stabilizations_;
      ScalarShapeFunctionSetType scalarShapeFunctionSet_;
      EdgeShapeFunctionSetType edgeShapeFunctionSet_;
      unsigned int polOrder_;
      bool conforming_;
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

      // want to iterate over each polygon separately
      std::vector< std::vector< ElementSeedType > > entitySeeds( agglomeration().size() );
      for( const ElementType &element : elements( static_cast< typename GridPart::GridViewType >( gridPart() ), Partitions::interiorBorder ) )
        entitySeeds[ agglomeration().index( element ) ].push_back( element.seed() );

      const std::size_t numShapeFunctions = scalarShapeFunctionSet_.size();
      const std::size_t numShapeFunctionsMinus1 =
              Dune::Fem::OrthonormalShapeFunctions< DomainType::dimension >::size(polOrder-1);
      const std::size_t numShapeFunctionsMinus2 = polOrder==1?0:
              Dune::Fem::OrthonormalShapeFunctions< DomainType::dimension >::size(polOrder-2);

      DynamicMatrix< DomainFieldType > D, C, Hp, HpMinus1;
      DynamicMatrix< DomainType > G, R;
      DynamicMatrix< DomainFieldType > edgePhi; // compute Phi_i on the edge

      LeftPseudoInverse< DomainFieldType > pseudoInverse( numShapeFunctions );

      valueProjections_.resize( agglomeration().size() );
      jacobianProjections_.resize( agglomeration().size() );
      stabilizations_.resize( agglomeration().size() );

      AgglomerationInterpolationType interpolation( blockMapper().indexSet(), polOrder, conforming_ );

      for( std::size_t agglomerate = 0; agglomerate < agglomeration().size(); ++agglomerate )
      {
        const auto &bbox = blockMapper_.indexSet().boundingBox(agglomerate);
        auto extend = bbox.second;
        extend -= bbox.first;

        const std::size_t numDofs = blockMapper().numDofs( agglomerate );

        D.resize( numDofs, numShapeFunctions, 0 );
        C.resize( numShapeFunctions, numDofs, 0 );
        Hp.resize( numShapeFunctions, numShapeFunctions, 0 );
        HpMinus1.resize( numShapeFunctionsMinus1, numShapeFunctionsMinus1, 0 );
        G.resize( numShapeFunctionsMinus1, numShapeFunctionsMinus1, DomainType(0) );
        R.resize( numShapeFunctionsMinus1, numDofs, DomainType(0) );

        for( const ElementSeedType &entitySeed : entitySeeds[ agglomerate ] )
        {
          const ElementType &element = gridPart().entity( entitySeed );
          const auto geometry = element.geometry();
          const auto &refElement = ReferenceElements< typename GridPart::ctype, GridPart::dimension >::general( element.type() );

          BoundingBoxBasisFunctionSet< GridPart, ScalarShapeFunctionSetType > shapeFunctionSet( element, bbox, scalarShapeFunctionSet_ );
          interpolation( shapeFunctionSet, D );

          if (polOrder > 1)
          {
            // compute mass matrices Hp, HpMinus1, and the gradient matrices G^l
            Fem::ElementQuadrature< GridPart, 0 > quadrature( element, 2*polOrder );
            for( std::size_t qp = 0; qp < quadrature.nop(); ++qp )
            {
              const DomainFieldType weight = geometry.integrationElement( quadrature.point( qp ) ) * quadrature.weight( qp );
              shapeFunctionSet.evaluateEach( quadrature[ qp ], [ & ] ( std::size_t alpha, FieldVector< DomainFieldType, 1 > phi ) {
                  shapeFunctionSet.evaluateEach( quadrature[ qp ], [ & ] ( std::size_t beta, FieldVector< DomainFieldType, 1 > psi ) {
                      if (alpha<numShapeFunctionsMinus1 &&
                          beta<numShapeFunctionsMinus1) // basis set is hierarchic so we can compute HpMinus1 using the order p shapeFunctionSet
                        HpMinus1[alpha][beta] += phi[0]*psi[0]*weight;
                      Hp[alpha][beta] += phi[0]*psi[0]*weight;
                    } );
                  if (alpha<numShapeFunctionsMinus1)
                    shapeFunctionSet.jacobianEach( quadrature[ qp ], [ & ] ( std::size_t beta, typename ScalarShapeFunctionSetType::JacobianRangeType psi ) {
                        // psi[0][0] /= extend[0]; // need correct scalling here
                        // psi[0][1] /= extend[1];
                        if (beta<numShapeFunctionsMinus1)
                          G[alpha][beta].axpy(phi[0]*weight, psi[0]);
                      } );
                } );
            } // quadrature loop
          } // polOrder > 1

          // compute the boundary terms for the gradient projection
          for( const auto &intersection : intersections( static_cast< typename GridPart::GridViewType >( gridPart() ), element ) )
          {
            if( !intersection.boundary() && (agglomeration().index( intersection.outside() ) == agglomerate) )
              continue;
            assert( intersection.conforming() );
            auto normal = intersection.centerUnitOuterNormal();
            std::vector<int> mask; // contains indices with Phi_mask[i] is attached to edge
            edgePhi.resize(edgeShapeFunctionSet_.size(),edgeShapeFunctionSet_.size(),0);
            interpolation( intersection, edgeShapeFunctionSet_, edgePhi, mask );
            edgePhi.invert();
            { // test edgePhi
              assert( mask.size() == edgeShapeFunctionSet_.size() );
              std::vector<double> lambda(numDofs);
#if 1 // terrible hack!
              bool succ = true;
              for (int i=0;i<mask.size();++i)
              {
                std::fill(lambda.begin(),lambda.end(),0);
                PhiEdge<GridPartType, Dune::DynamicMatrix<double>,EdgeShapeFunctionSetType>
                  phiEdge(gridPart(),intersection,edgePhi,edgeShapeFunctionSet_,i); // behaves like Phi_mask[i] restricted to edge
                interpolation(element,phiEdge,lambda);
                for (int k=0;k<numDofs;++k) // lambda should be 1 for k=mask[i] otherwise 0
                  succ &= ( mask[i]==k? std::abs(lambda[k]-1)<1e-10: std::abs(lambda[k])<1e-10 );
              }
              if (!succ) std::swap(mask[0],mask[1]);
#endif
              for (int i=0;i<mask.size();++i)
              {
                std::fill(lambda.begin(),lambda.end(),0);
                PhiEdge<GridPartType, Dune::DynamicMatrix<double>,EdgeShapeFunctionSetType>
                  phiEdge(gridPart(),intersection,edgePhi,edgeShapeFunctionSet_,i); // behaves like Phi_mask[i] restricted to edge
                interpolation(element,phiEdge,lambda);
                for (int k=0;k<numDofs;++k) // lambda should be 1 for k=mask[i] otherwise 0
                  assert( mask[i]==k? std::abs(lambda[k]-1)<1e-10: std::abs(lambda[k])<1e-10 );
              }
            }
            // now compute int_e Phi_mask[i] m_alpha
            typedef Fem::ElementQuadrature< GridPart, 1 > EdgeQuadratureType;
            EdgeQuadratureType quadrature( gridPart(), intersection, 2*polOrder-1, EdgeQuadratureType::INSIDE );
            for( std::size_t qp = 0; qp < quadrature.nop(); ++qp )
            {
              auto x = quadrature.localPoint(qp);
              auto y = intersection.geometryInInside().global(x);
              const DomainFieldType weight = intersection.geometry().integrationElement( x ) * quadrature.weight( qp );
              shapeFunctionSet.evaluateEach( y, [ & ] ( std::size_t alpha, FieldVector< DomainFieldType, 1 > phi ) {
                 if (alpha<numShapeFunctionsMinus1)
                    edgeShapeFunctionSet_.evaluateEach( x, [ & ] ( std::size_t beta, FieldVector< DomainFieldType, 1 > psi ) {
                        for (int s=0;s<mask.size();++s) // note that edgePhi is the transposed of the basis transform matrix
                          R[alpha][mask[s]].axpy( edgePhi[beta][s]*psi[0]*phi[0]*weight, normal);
                    } );
              } );
            } // quadrature loop
          } // loop over intersections
        } // loop over triangles in agglomerate
        // finished agglomerating all auxiliary matrices
        // now compute projection matrices and stabilization

        // volume
        DomainFieldType H0 = blockMapper_.indexSet().volume(agglomerate);

        auto &valueProjection    = valueProjections_[ agglomerate ];
        auto &jacobianProjection = jacobianProjections_[ agglomerate ];
        jacobianProjection.resize( numShapeFunctions );
        for( std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha )
          jacobianProjection[ alpha ].resize( numDofs, DomainType( 0 ) );

        pseudoInverse( D, valueProjection );

        if (polOrder > 1)
        {
          // modify C for inner dofs
          std::size_t alpha=0;
          for (; alpha<numShapeFunctionsMinus2; ++alpha)
            C[alpha][alpha+numDofs-numShapeFunctionsMinus2] = H0;
          for (; alpha<numShapeFunctions; ++alpha)
            for (std::size_t i=0; i<numDofs; ++i)
              for (std::size_t beta=0; beta<numShapeFunctions; ++beta)
                C[alpha][i] += Hp[alpha][beta]*valueProjection[beta][i];

          Hp.invert();
          HpMinus1.invert();

          auto Gtmp = G;
          for (std::size_t alpha=0; alpha<numShapeFunctionsMinus1; ++alpha)
          {
            for (std::size_t beta=0; beta<numShapeFunctionsMinus1; ++beta)
            {
              G[alpha][beta] = DomainType(0);
              for (std::size_t gamma=0; gamma<numShapeFunctionsMinus1; ++gamma)
                G[alpha][beta].axpy(HpMinus1[alpha][gamma],Gtmp[gamma][beta]);
            }
          }

          { // test G matrix
            for( const ElementSeedType &entitySeed : entitySeeds[ agglomerate ] )
            {
              const ElementType &element = gridPart().entity( entitySeed );
              BoundingBoxBasisFunctionSet< GridPart, ScalarShapeFunctionSetType > shapeFunctionSet( element, bbox, scalarShapeFunctionSet_ );
              Fem::ElementQuadrature< GridPart, 0 > quad( element, 2*polOrder );
              for( std::size_t qp = 0; qp < quad.nop(); ++qp )
              {
                shapeFunctionSet.jacobianEach( quad[qp], [ & ] ( std::size_t alpha, FieldMatrix< DomainFieldType, 1,2 > phi ) {
                  if (alpha<numShapeFunctionsMinus1)
                  {
                    Dune::FieldVector<double,2> d;
                    Derivative<GridPartType, Dune::DynamicMatrix<DomainType>,decltype(shapeFunctionSet)>
                        derivative(gridPart(),G,shapeFunctionSet,alpha); // used G matrix to compute gradients of monomials
                    derivative.evaluate(quad[qp],d);
                    d -= phi[0];
                    assert(d.two_norm() < 1e-10);
                  }
                });
              }
            }
          } // test G matrix

          // add interior integrals for gradient projection
          for (std::size_t alpha=0; alpha<numShapeFunctionsMinus1; ++alpha)
            for (std::size_t beta=0; beta<numShapeFunctionsMinus2; ++beta)
              R[alpha][beta+numDofs-numShapeFunctionsMinus2].axpy(-H0, G[beta][alpha]);

          // now compute projection by multiplying with inverse mass matrix
          for (std::size_t alpha=0; alpha<numShapeFunctions; ++alpha)
          {
            for (std::size_t i=0; i<numDofs; ++i)
            {
              valueProjection[alpha][i] = 0;
              for (std::size_t beta=0; beta<numShapeFunctions; ++beta)
                valueProjection[alpha][i] += Hp[alpha][beta]*C[beta][i];
              if (alpha<numShapeFunctionsMinus1)
                for (std::size_t beta=0; beta<numShapeFunctionsMinus1; ++beta)
                  jacobianProjection[alpha][i].axpy(HpMinus1[alpha][beta],R[beta][i]);
            }
          }
        } // polOrder> 0
        else
        { // for p=1 we didn't compute the inverse of the mass matrix H_{p-1} -
          // it's simply 1/H0 so we implement the p=1 case separately
          for (std::size_t i=0; i<numDofs; ++i)
            jacobianProjection[0][i].axpy(1./H0, R[0][i]);
        }

#if 0
        // compute energy norm stability scalling
        std::vector<double> stabScaling(numDofs, 0);
        std::vector<typename BasisFunctionSetType::JacobianRangeType> dphi(numDofs);
        for( const ElementSeedType &entitySeed : entitySeeds[ agglomerate ] )
        {
          const ElementType &element = gridPart().entity( entitySeed );
          const auto geometry = element.geometry();
          const BasisFunctionSetType vemBaseSet = basisFunctionSet ( element );
          Fem::ElementQuadrature< GridPart, 0 > quadrature( element, 2*polOrder );
          for( std::size_t qp = 0; qp < quadrature.nop(); ++qp )
          {
            vemBaseSet.jacobianAll(quadrature[qp], dphi);
            const auto &x = quadrature.point(qp);
            const double weight = quadrature.weight(qp) * geometry.integrationElement(x);
            for (int i=0;i<numDofs;++i)
              stabScaling[i] += weight*(dphi[i][0]*dphi[i][0]);
          }
        }
        std::cout << "stabscaling= " << std::flush;
        for( std::size_t i = 0; i < numDofs; ++i )
          std::cout << stabScaling[i] << " ";
        std::cout << std::endl;
#endif
        // stabilization matrix
        Stabilization S( numDofs, numDofs, 0 );
        for( std::size_t i = 0; i < numDofs; ++i )
          S[ i ][ i ] = DomainFieldType( 1 );
        for( std::size_t i = 0; i < numDofs; ++i )
          for( std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha )
            for( std::size_t j = 0; j < numDofs; ++j )
              S[ i ][ j ] -= D[ i ][ alpha ] * valueProjection[ alpha ][ j ];
        Stabilization &stabilization = stabilizations_[ agglomerate ];
        stabilization.resize( numDofs, numDofs, 0 );
        for( std::size_t i = 0; i < numDofs; ++i )
          for( std::size_t j = 0; j < numDofs; ++j )
            for( std::size_t k = 0; k < numDofs; ++k )
              stabilization[ i ][ j ] += S[ k ][ i ] * S[ k ][ j ];
              // stabilization[ i ][ j ] += S[ k ][ i ] * std::max(1.,stabScaling[k]) * S[ k ][ j ];
      }  // loop over agglomerates
    } // build projections

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

#if 1 // for interpolation
    namespace Capabilities
    {
      template< class FunctionSpace, class GridPart, int polOrder >
      struct hasInterpolation< Vem::AgglomerationVEMSpace< FunctionSpace, GridPart, polOrder > >
      {
        static const bool v = false;
      };
    }
#endif
  } // namespace Fem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_AGGLOMERATION_HH
