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

#include <dune/vem/space/indexset.hh>
#include <dune/vem/agglomeration/dofmapper.hh>
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
      typedef Fem::VectorialBasisFunctionSet< ScalarBasisFunctionSetType, typename FunctionSpaceType::RangeType > BasisFunctionSetType;

      typedef Hybrid::IndexRange< int, FunctionSpaceType::dimRange > LocalBlockIndices;
      typedef VemAgglomerationIndexSet< GridPartType > AgglomerationIndexSetType;
      typedef AgglomerationDofMapper< GridPartType, AgglomerationIndexSetType > BlockMapperType;

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

      typedef typename Traits::AgglomerationIndexSetType AgglomerationIndexSetType;
      typedef AgglomerationVEMInterpolation< AgglomerationIndexSetType > AgglomerationInterpolationType;
      typedef typename Traits::ScalarShapeFunctionSetType ScalarShapeFunctionSetType;

    public:
      typedef typename BaseType::BasisFunctionSetType BasisFunctionSetType;

      typedef typename BaseType::BlockMapperType BlockMapperType;

      typedef typename BaseType::EntityType EntityType;
      typedef typename BaseType::GridPartType GridPartType;
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld-1,1> EdgeFSType;
      typedef Dune::Fem::OrthonormalShapeFunctionSet<EdgeFSType> EdgeShapeFunctionSetType;
      typedef typename BasisFunctionSetType::DomainFieldType DomainFieldType;
      typedef typename BasisFunctionSetType::DomainType DomainType;
      typedef typename GridPart::template Codim< 0 >::EntityType ElementType;
      typedef typename GridPart::template Codim< 0 >::EntitySeedType ElementSeedType;


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

      // basisChoice:
      // 1: use onb for inner moments but not for computing projections
      // 2: use onb for both the inner moments and computation of projection
      // 3: don't use onb at all
      AgglomerationVEMSpace ( AgglomerationType &agglomeration, std::vector<int> testSpaces,
          int basisChoice)
        : BaseType( agglomeration.gridPart() ),
          agIndexSet_( agglomeration, testSpaces ),
          blockMapper_( agIndexSet_, agIndexSet_.dofsPerCodim() ),
          interpolation_(blockMapper().indexSet(), polOrder, basisChoice!=3 ),
          scalarShapeFunctionSet_( Dune::GeometryType( Dune::GeometryType::cube, GridPart::dimension ) ),
          edgeShapeFunctionSet_(   Dune::GeometryType( Dune::GeometryType::cube, GridPart::dimension-1 ),
              testSpaces[1] +       // edge order
                (testSpaces[0]+1)*2 // vertex order * number of vertices on edge
              ),
          polOrder_( polOrder ),
          useOnb_(basisChoice==2)
      {
#if 0
        const int innerTestSpace = testSpaces[2];
        assert(innerTestSpace>=-1);
        const std::size_t numShapeFunctions = scalarShapeFunctionSet_.size(); // uses polOrder
        // const std::size_t  //! this casuses a weird internal compiler error...
        int numGradShapeFunctions =
                Dune::Fem::OrthonormalShapeFunctions< DomainType::dimension >::size(innerTestSpace+1);
        const std::size_t numInnerShapeFunctions = innerTestSpace<0?0:
                Dune::Fem::OrthonormalShapeFunctions< DomainType::dimension >::size(innerTestSpace);
        std::cout << "******************************************\n";
        std::cout << "AgglomerationVEMSpace: "
          << polOrder << " (" << testSpaces[0] << "," << testSpaces[1] << "," << testSpaces[2] << ") "
          << "inner:" << numInnerShapeFunctions << "    "
          << "edge:" << edgeShapeFunctionSet_.size() << " " << "value:" << scalarShapeFunctionSet_.size() << " "
          << "grad:" << numGradShapeFunctions
          << std::endl;
        std::cout << "******************************************\n";
#endif
        onbBasis( agglomeration, scalarShapeFunctionSet_, agIndexSet_.boundingBox() );
        buildProjections();
      }

      const BasisFunctionSetType basisFunctionSet ( const EntityType &entity ) const
      {
        const std::size_t agglomerate = agglomeration().index( entity );
        const auto &valueProjection = valueProjections_[ agglomerate ];
        const auto &jacobianProjection = jacobianProjections_[ agglomerate ];
        const auto &bbox = blockMapper_.indexSet().boundingBox(agglomerate);
        // scalar ONB Basis proxy
        typename Traits::DGTraitsType::ShapeFunctionSetType scalarShapeFunctionSet( &scalarShapeFunctionSet_ );
        // scalar BB Basis
        typename Traits::ScalarBBBasisFunctionSetType bbScalarBasisFunctionSet( entity, bbox,
            useOnb_, std::move( scalarShapeFunctionSet ) );
        // vectorial extended VEM Basis
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

      AgglomerationInterpolationType interpolation() const
      {
        return interpolation_;
      }

    private:
      void buildProjections ();

      // issue with making these const: use of delete default constructor in some python bindings...
      AgglomerationIndexSetType agIndexSet_;
      mutable BlockMapperType blockMapper_;
      AgglomerationInterpolationType interpolation_;
      std::vector< typename Traits::ScalarBasisFunctionSetType::ValueProjection > valueProjections_;
      std::vector< typename Traits::ScalarBasisFunctionSetType::JacobianProjection > jacobianProjections_;
      std::vector< Stabilization > stabilizations_;
      ScalarShapeFunctionSetType scalarShapeFunctionSet_;
      EdgeShapeFunctionSetType edgeShapeFunctionSet_;
      unsigned int polOrder_;
      const bool useOnb_;
    };


    // Implementation of AgglomerationVEMSpace
    // ---------------------------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    inline void AgglomerationVEMSpace< FunctionSpace, GridPart, polOrder >::buildProjections ()
    {
      // want to iterate over each polygon separately - so collect all
      // triangles from a given polygon
      std::vector< std::vector< ElementSeedType > > entitySeeds( agglomeration().size() );
      for( const ElementType &element : elements( static_cast< typename GridPart::GridViewType >( gridPart() ), Partitions::interiorBorder ) )
        entitySeeds[ agglomeration().index( element ) ].push_back( element.seed() );

      // get polynomial degree of inner test space and then compute
      // - size of space for value projection
      // - size of space for gradient projection
      // - size of space for inner moments
      const int innerTestSpace = agIndexSet_.testSpaces()[2];
      assert(innerTestSpace>=-1);
      const std::size_t numShapeFunctions = scalarShapeFunctionSet_.size(); // uses polOrder
      // const std::size_t  //! this casuses a weird internal compiler error...
      int numGradShapeFunctions =
              Dune::Fem::OrthonormalShapeFunctions< DomainType::dimension >::size(innerTestSpace+1);
      const std::size_t numInnerShapeFunctions = innerTestSpace<0?0:
              Dune::Fem::OrthonormalShapeFunctions< DomainType::dimension >::size(innerTestSpace);

      // set up matrices used for constructing gradient, value, and edge projections
      // Note: the code is set up with the assumption that the dofs suffice to compute the edge projection
      DynamicMatrix< DomainFieldType > D, C, Hp, HpGrad, HpInv, HpGradInv;
      DynamicMatrix< DomainType > G, R;
      DynamicMatrix< DomainFieldType > edgePhi;

      LeftPseudoInverse< DomainFieldType > pseudoInverse( numShapeFunctions );

      // these are the matrices we need to compute
      valueProjections_.resize( agglomeration().size() );
      jacobianProjections_.resize( agglomeration().size() );
      stabilizations_.resize( agglomeration().size() );

      // start iteration over all polygons
      for( std::size_t agglomerate = 0; agglomerate < agglomeration().size(); ++agglomerate )
      {
        const auto &bbox = blockMapper_.indexSet().boundingBox(agglomerate);
        const std::size_t numDofs = blockMapper().numDofs( agglomerate );

        D.resize( numDofs, numShapeFunctions, 0 );
        C.resize( numShapeFunctions, numDofs, 0 );
        Hp.resize( numShapeFunctions, numShapeFunctions, 0 );
        HpGrad.resize( numGradShapeFunctions, numGradShapeFunctions, 0 );
        G.resize( numGradShapeFunctions, numGradShapeFunctions, DomainType(0) );
        R.resize( numGradShapeFunctions, numDofs, DomainType(0) );

        // iterate over the triangles of this polygon
        for( const ElementSeedType &entitySeed : entitySeeds[ agglomerate ] )
        {
          const ElementType &element = gridPart().entity( entitySeed );
          const auto geometry = element.geometry();
          const auto &refElement = ReferenceElements< typename GridPart::ctype, GridPart::dimension >::general( element.type() );

          // get the bounding box monomials and apply all dofs to them
          BoundingBoxBasisFunctionSet< GridPart, ScalarShapeFunctionSetType > shapeFunctionSet( element, bbox,
              useOnb_, scalarShapeFunctionSet_ );
          interpolation_( shapeFunctionSet, D );

          // compute mass matrices Hp, HpGrad, and the gradient matrices G^l
          Fem::ElementQuadrature< GridPart, 0 > quadrature( element, 2*polOrder );
          for( std::size_t qp = 0; qp < quadrature.nop(); ++qp )
          {
            const DomainFieldType weight = geometry.integrationElement( quadrature.point( qp ) ) * quadrature.weight( qp );
            shapeFunctionSet.evaluateEach( quadrature[ qp ], [ & ] ( std::size_t alpha, FieldVector< DomainFieldType, 1 > phi ) {
                shapeFunctionSet.evaluateEach( quadrature[ qp ], [ & ] ( std::size_t beta, FieldVector< DomainFieldType, 1 > psi ) {
                    Hp[alpha][beta] += phi[0]*psi[0]*weight;
                    if (alpha<numGradShapeFunctions &&
                        beta<numGradShapeFunctions) // basis set is hierarchic so we can compute HpGrad using the order p shapeFunctionSet
                      HpGrad[alpha][beta] += phi[0]*psi[0]*weight;
                  } );
                if (alpha<numGradShapeFunctions)
                  shapeFunctionSet.jacobianEach( quadrature[ qp ], [ & ] ( std::size_t beta, typename ScalarShapeFunctionSetType::JacobianRangeType psi ) {
                      if (beta<numGradShapeFunctions)
                        G[alpha][beta].axpy(phi[0]*weight, psi[0]);
                    } );
              } );
          } // quadrature loop

          // compute the boundary terms for the gradient projection
          for( const auto &intersection : intersections( static_cast< typename GridPart::GridViewType >( gridPart() ), element ) )
          {
            // ignore edges inside the given polygon
            if( !intersection.boundary() && (agglomeration().index( intersection.outside() ) == agglomerate) )
              continue;
            assert( intersection.conforming() );
            auto normal = intersection.centerUnitOuterNormal();
            std::vector<int> mask; // contains indices with Phi_mask[i] is attached to given edge
            edgePhi.resize(edgeShapeFunctionSet_.size(),edgeShapeFunctionSet_.size(),0);
            interpolation_( intersection, edgeShapeFunctionSet_, edgePhi, mask );
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
                interpolation_(element,phiEdge,lambda);
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
                interpolation_(element,phiEdge,lambda);
                for (int k=0;k<numDofs;++k) // lambda should be 1 for k=mask[i] otherwise 0
                  assert( mask[i]==k? std::abs(lambda[k]-1)<1e-10: std::abs(lambda[k])<1e-10 );
              }
            } // testing

            // now compute int_e Phi_mask[i] m_alpha
            typedef Fem::ElementQuadrature< GridPart, 1 > EdgeQuadratureType;
            EdgeQuadratureType quadrature( gridPart(), intersection, 2*polOrder, EdgeQuadratureType::INSIDE );
            for( std::size_t qp = 0; qp < quadrature.nop(); ++qp )
            {
              auto x = quadrature.localPoint(qp);
              auto y = intersection.geometryInInside().global(x);
              const DomainFieldType weight = intersection.geometry().integrationElement( x ) * quadrature.weight( qp );
              shapeFunctionSet.evaluateEach( y, [ & ] ( std::size_t alpha, FieldVector< DomainFieldType, 1 > phi ) {
                 if (alpha<numGradShapeFunctions)
                    edgeShapeFunctionSet_.evaluateEach( x, [ & ] ( std::size_t beta, FieldVector< DomainFieldType, 1 > psi ) {
                        for (int s=0;s<mask.size();++s) // note that edgePhi is the transposed of the basis transform matrix
                          R[alpha][mask[s]].axpy( edgePhi[beta][s]*psi[0]*phi[0]*weight, normal);
                    } );
              } );
            } // quadrature loop
          } // loop over intersections
        } // loop over triangles in agglomerate

        // finished agglomerating all auxiliary matrices
        // now compute projection matrices and stabilization for the given polygon

        // volume of polygon
        DomainFieldType H0 = blockMapper_.indexSet().volume(agglomerate);

        auto &valueProjection    = valueProjections_[ agglomerate ];
        auto &jacobianProjection = jacobianProjections_[ agglomerate ];
        jacobianProjection.resize( numShapeFunctions );
        for( std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha )
          jacobianProjection[ alpha ].resize( numDofs, DomainType( 0 ) );

        pseudoInverse( D, valueProjection );

        if (numInnerShapeFunctions > 0)
        {
          // modify C for inner dofs
          std::size_t alpha=0;
          for (; alpha<numInnerShapeFunctions; ++alpha)
            C[alpha][alpha+numDofs-numInnerShapeFunctions] = H0;
          for (; alpha<numShapeFunctions; ++alpha)
            for (std::size_t i=0; i<numDofs; ++i)
              for (std::size_t beta=0; beta<numShapeFunctions; ++beta)
                C[alpha][i] += Hp[alpha][beta]*valueProjection[beta][i];

          HpInv = Hp;
          HpInv.invert();
          HpGradInv = HpGrad;
          HpGradInv.invert();

          auto Gtmp = G;
          for (std::size_t alpha=0; alpha<numGradShapeFunctions; ++alpha)
          {
            for (std::size_t beta=0; beta<numGradShapeFunctions; ++beta)
            {
              G[alpha][beta] = DomainType(0);
              for (std::size_t gamma=0; gamma<numGradShapeFunctions; ++gamma)
                G[alpha][beta].axpy(HpGradInv[alpha][gamma],Gtmp[gamma][beta]);
            }
          }

          { // test G matrix
            for( const ElementSeedType &entitySeed : entitySeeds[ agglomerate ] )
            {
              const ElementType &element = gridPart().entity( entitySeed );
              BoundingBoxBasisFunctionSet< GridPart, ScalarShapeFunctionSetType > shapeFunctionSet( element, bbox,
                  useOnb_, scalarShapeFunctionSet_ );
              Fem::ElementQuadrature< GridPart, 0 > quad( element, 2*polOrder );
              for( std::size_t qp = 0; qp < quad.nop(); ++qp )
              {
                shapeFunctionSet.jacobianEach( quad[qp], [ & ] ( std::size_t alpha, FieldMatrix< DomainFieldType, 1,2 > phi ) {
                  if (alpha<numGradShapeFunctions)
                  {
                    Dune::FieldVector<double,2> d;
                    Derivative<GridPartType, Dune::DynamicMatrix<DomainType>,decltype(shapeFunctionSet)>
                        derivative(gridPart(),G,shapeFunctionSet,alpha); // used G matrix to compute gradients of monomials
                    derivative.evaluate(quad[qp],d);
                    d -= phi[0];
                    assert(d.two_norm() < 1e-8);
                  }
                });
              }
            }
          } // test G matrix

          // add interior integrals for gradient projection
          for (std::size_t alpha=0; alpha<numGradShapeFunctions; ++alpha)
            for (std::size_t beta=0; beta<numInnerShapeFunctions; ++beta)
              R[alpha][beta+numDofs-numInnerShapeFunctions].axpy(-H0, G[beta][alpha]);

          // now compute projection by multiplying with inverse mass matrix
          for (std::size_t alpha=0; alpha<numShapeFunctions; ++alpha)
          {
            for (std::size_t i=0; i<numDofs; ++i)
            {
              valueProjection[alpha][i] = 0;
              for (std::size_t beta=0; beta<numShapeFunctions; ++beta)
                valueProjection[alpha][i] += HpInv[alpha][beta]*C[beta][i];
              if (alpha<numGradShapeFunctions)
                for (std::size_t beta=0; beta<numGradShapeFunctions; ++beta)
                  jacobianProjection[alpha][i].axpy(HpGradInv[alpha][beta],R[beta][i]);
            }
          }
        } // have some inner moments
        else
        { // with no inner moments we didn't need to compute the inverse of the mass matrix H_{p-1} -
          // it's simply 1/H0 so we implement this case separately
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
#if 1
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
#else
        Stabilization &stabilization = stabilizations_[ agglomerate ];
        stabilization.resize( numDofs, numDofs, 0 );
        for( std::size_t i = 0; i < numDofs; ++i )
          stabilization[ i ][ i ] = DomainFieldType( 1 );
        for( std::size_t i = 0; i < numDofs; ++i )
          for( std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha )
            for( std::size_t j = 0; j < numDofs; ++j )
              stabilization[ i ][ j ] -= D[ i ][ alpha ] * valueProjection[ alpha ][ j ];
#endif


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

    namespace Capabilities
    {
      template< class FunctionSpace, class GridPart, int polOrder >
      struct hasInterpolation< Vem::AgglomerationVEMSpace< FunctionSpace, GridPart, polOrder > >
      {
        static const bool v = false;
      };
    }

  } // namespace Fem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_AGGLOMERATION_HH
