#ifndef DUNE_VEM_OPERATOR_ELLIPTIC_HH
#define DUNE_VEM_OPERATOR_ELLIPTIC_HH

#include <cstddef>

#include <dune/common/fmatrix.hh>

#include <dune/fem/operator/common/operator.hh>
#include <dune/fem/operator/common/stencil.hh>
#include <dune/fem/quadrature/cachingquadrature.hh>

#include <dune/fem/operator/common/differentiableoperator.hh>

#include <dune/vem/operator/constraints/dirichlet.hh>


namespace Dune
{

  namespace Vem
  {

    // EllipticOperator
    // ----------------

    template< class DomainDiscreteFunction, class RangeDiscreteFunction, class Model,
              class Constraints = Dune::Vem::DirichletConstraints< Model, typename RangeDiscreteFunction::DiscreteFunctionSpaceType > >
    struct VEMEllipticOperator
      : public virtual Dune::Fem::Operator< DomainDiscreteFunction, RangeDiscreteFunction >
    {
    protected:
      typedef DomainDiscreteFunction DomainDiscreteFunctionType;
      typedef RangeDiscreteFunction RangeDiscreteFunctionType;
      typedef Model ModelType;
      typedef Constraints ConstraintsType;

      typedef typename DomainDiscreteFunctionType::DiscreteFunctionSpaceType DomainDiscreteFunctionSpaceType;
      typedef typename DomainDiscreteFunctionType::LocalFunctionType DomainLocalFunctionType;
      typedef typename DomainLocalFunctionType::RangeType DomainRangeType;
      typedef typename DomainLocalFunctionType::JacobianRangeType DomainJacobianRangeType;
      typedef typename RangeDiscreteFunctionType::DiscreteFunctionSpaceType RangeDiscreteFunctionSpaceType;
      typedef typename RangeDiscreteFunctionType::LocalFunctionType RangeLocalFunctionType;
      typedef typename RangeLocalFunctionType::RangeType RangeRangeType;
      typedef typename RangeLocalFunctionType::JacobianRangeType RangeJacobianRangeType;

      typedef typename RangeDiscreteFunctionSpaceType::IteratorType IteratorType;
      typedef typename IteratorType::Entity EntityType;
      typedef typename EntityType::Geometry GeometryType;
      typedef typename RangeDiscreteFunctionSpaceType::DomainType DomainType;
      typedef typename RangeDiscreteFunctionSpaceType::GridPartType GridPartType;

      typedef Dune::Fem::CachingQuadrature< GridPartType, 0 > QuadratureType;
      typedef Dune::Fem::ElementQuadrature< GridPartType, 1 > FaceQuadratureType;

    public:
      VEMEllipticOperator ( const ModelType &model, const RangeDiscreteFunctionSpaceType &rangeSpace )
        : model_( model ), constraints_( model, rangeSpace )
      {}

      virtual void operator() ( const DomainDiscreteFunctionType &u, RangeDiscreteFunctionType &w ) const;

      const ModelType &model () const { return model_; }
      const ConstraintsType &constraints () const { return constraints_; }

    private:
      ModelType model_;
      ConstraintsType constraints_;
    };



    // DifferentiableEllipticOperator
    // ------------------------------

    template< class JacobianOperator, class Model,
              class Constraints = Dune::Vem::DirichletConstraints< Model, typename JacobianOperator::RangeFunctionType::DiscreteFunctionSpaceType > >
    struct DifferentiableVEMEllipticOperator
      : public VEMEllipticOperator< typename JacobianOperator::DomainFunctionType, typename JacobianOperator::RangeFunctionType, Model, Constraints >,
        public Dune::Fem::DifferentiableOperator< JacobianOperator >
    {
      typedef VEMEllipticOperator< typename JacobianOperator::DomainFunctionType, typename JacobianOperator::RangeFunctionType, Model, Constraints > BaseType;

      typedef JacobianOperator JacobianOperatorType;

      typedef typename BaseType::DomainDiscreteFunctionType DomainDiscreteFunctionType;
      typedef typename BaseType::RangeDiscreteFunctionType RangeDiscreteFunctionType;
      typedef typename BaseType::ModelType ModelType;

    protected:
      typedef typename DomainDiscreteFunctionType::DiscreteFunctionSpaceType DomainDiscreteFunctionSpaceType;
      typedef typename DomainDiscreteFunctionType::LocalFunctionType DomainLocalFunctionType;
      typedef typename DomainLocalFunctionType::RangeType DomainRangeType;
      typedef typename DomainLocalFunctionType::JacobianRangeType DomainJacobianRangeType;
      typedef typename RangeDiscreteFunctionType::DiscreteFunctionSpaceType RangeDiscreteFunctionSpaceType;
      typedef typename RangeDiscreteFunctionType::LocalFunctionType RangeLocalFunctionType;
      typedef typename RangeLocalFunctionType::RangeType RangeRangeType;
      typedef typename RangeLocalFunctionType::JacobianRangeType RangeJacobianRangeType;

      typedef typename RangeDiscreteFunctionSpaceType::IteratorType IteratorType;
      typedef typename IteratorType::Entity EntityType;
      typedef typename EntityType::Geometry GeometryType;
      typedef typename RangeDiscreteFunctionSpaceType::DomainType DomainType;
      typedef typename RangeDiscreteFunctionSpaceType::GridPartType GridPartType;

      typedef typename BaseType::QuadratureType QuadratureType;
      typedef typename BaseType::FaceQuadratureType FaceQuadratureType;

    public:
      DifferentiableVEMEllipticOperator ( const ModelType &model, const RangeDiscreteFunctionSpaceType &space )
        : BaseType( model, space )
      {}

      void jacobian ( const DomainDiscreteFunctionType &u, JacobianOperatorType &jOp ) const;

    protected:
      using BaseType::model;
      using BaseType::constraints;
    };



    // Implementation of EllipticOperator
    // ----------------------------------

    template< class DomainDiscreteFunction, class RangeDiscreteFunction, class Model, class Constraints >
    void VEMEllipticOperator< DomainDiscreteFunction, RangeDiscreteFunction, Model, Constraints >
    ::operator() ( const DomainDiscreteFunctionType &u, RangeDiscreteFunctionType &w ) const
    {
      w.clear();
      const RangeDiscreteFunctionSpaceType &dfSpace = w.space();

      std::vector< bool > stabilization( dfSpace.agglomeration().size(), false );

      const GridPartType &gridPart = w.gridPart();
      for( const auto &entity : Dune::elements( static_cast< typename GridPartType::GridViewType >( gridPart ), Dune::Partitions::interiorBorder ) )
      {
        const GeometryType &geometry = entity.geometry();

        const DomainLocalFunctionType uLocal = u.localFunction( entity );
        RangeLocalFunctionType wLocal = w.localFunction( entity );

        if( !stabilization[ dfSpace.agglomeration().index( entity ) ] )
        {
          const auto &stabMatrix = dfSpace.stabilization( entity );
          for( std::size_t r = 0; r < stabMatrix.rows(); ++r )
            for( std::size_t c = 0; c < stabMatrix.cols(); ++c )
              wLocal[ r ] += stabMatrix[ r ][ c ] * uLocal[ c ];
          stabilization[ dfSpace.agglomeration().index( entity ) ] = true;
        }

        const int quadOrder = uLocal.order() + wLocal.order();

        {
          QuadratureType quadrature( entity, quadOrder );
          const std::size_t numQuadraturePoints = quadrature.nop();
          for( std::size_t pt = 0; pt < numQuadraturePoints; ++pt )
          {
            const typename QuadratureType::CoordinateType &x = quadrature.point( pt );
            const double weight = quadrature.weight( pt ) * geometry.integrationElement( x );

            DomainRangeType vu;
            uLocal.evaluate( quadrature[ pt ], vu );
            DomainJacobianRangeType du;
            uLocal.jacobian( quadrature[ pt ], du );

            RangeRangeType avu( 0 );
            model().source( entity, quadrature[ pt ], vu, du, avu );
            avu *= weight;

            RangeJacobianRangeType adu( 0 );
            model().diffusiveFlux( entity, quadrature[ pt ], vu, du, adu );
            adu *= weight;

            wLocal.axpy( quadrature[ pt ], avu, adu );
          }
        }

        if( model().hasNeumanBoundary() && entity.hasBoundaryIntersections() )
        {
          for( const auto &intersection : Dune::intersections( static_cast< typename GridPartType::GridViewType >( gridPart ), entity ) )
          {
            if( !intersection.boundary() )
              continue;

            Dune::FieldVector< bool, RangeRangeType::dimension > components( true );
            bool hasDirichletComponent = model().isDirichletIntersection( intersection, components );

            const auto intersectionGeometry = intersection.geometry();
            FaceQuadratureType quadInside( dfSpace.gridPart(), intersection, quadOrder, FaceQuadratureType::INSIDE );
            const std::size_t numQuadraturePoints = quadInside.nop();
            for( std::size_t pt = 0; pt < numQuadraturePoints; ++pt )
            {
              const typename FaceQuadratureType::LocalCoordinateType &x = quadInside.localPoint( pt );
              double weight = quadInside.weight( pt ) * intersectionGeometry.integrationElement( x );
              DomainRangeType vu;
              uLocal.evaluate( quadInside[ pt ], vu );
              RangeRangeType alpha( 0 );
              model().alpha( entity, quadInside[ pt ], vu, alpha );
              alpha *= weight;
              for( int k = 0; k < RangeRangeType::dimension; ++k )
                if( hasDirichletComponent && components[ k ] )
                  alpha[ k ] = 0;
              wLocal.axpy( quadInside[ pt ], alpha );
            }
          }
        }
      }

      w.communicate();

      constraints()( u, w );
    }



    // Implementation of DifferentiableEllipticOperator
    // ------------------------------------------------

    template< class JacobianOperator, class Model, class Constraints >
    void DifferentiableVEMEllipticOperator< JacobianOperator, Model, Constraints >
    ::jacobian ( const DomainDiscreteFunctionType &u, JacobianOperator &jOp ) const
    {
      typedef typename JacobianOperator::LocalMatrixType LocalMatrixType;
      typedef typename DomainDiscreteFunctionSpaceType::BasisFunctionSetType DomainBasisFunctionSetType;
      typedef typename RangeDiscreteFunctionSpaceType::BasisFunctionSetType RangeBasisFunctionSetType;

      const DomainDiscreteFunctionSpaceType &domainSpace = jOp.domainSpace();
      const RangeDiscreteFunctionSpaceType  &rangeSpace = jOp.rangeSpace();

      Dune::Fem::DiagonalStencil< DomainDiscreteFunctionSpaceType, RangeDiscreteFunctionSpaceType >
      stencil( domainSpace, rangeSpace );
      jOp.reserve( stencil );
      jOp.clear();

      const int domainBlockSize = domainSpace.localBlockSize; // is equal to 1 for scalar functions
      std::vector< typename DomainLocalFunctionType::RangeType >         phi( domainSpace.blockMapper().maxNumDofs()*domainBlockSize );
      std::vector< typename DomainLocalFunctionType::JacobianRangeType > dphi( domainSpace.blockMapper().maxNumDofs()*domainBlockSize );
      const int rangeBlockSize = rangeSpace.localBlockSize; // is equal to 1 for scalar functions
      std::vector< typename RangeLocalFunctionType::RangeType >         rphi( rangeSpace.blockMapper().maxNumDofs()*rangeBlockSize );
      std::vector< typename RangeLocalFunctionType::JacobianRangeType > rdphi( rangeSpace.blockMapper().maxNumDofs()*rangeBlockSize );

      std::vector< bool > stabilization( rangeSpace.agglomeration().size(), false );

      const GridPartType &gridPart = rangeSpace.gridPart();
      for( const auto &entity : Dune::elements( static_cast< typename GridPartType::GridViewType >( gridPart ), Dune::Partitions::interiorBorder ) )
      {
        const GeometryType &geometry = entity.geometry();

        const DomainLocalFunctionType uLocal = u.localFunction( entity );
        LocalMatrixType jLocal = jOp.localMatrix( entity, entity );

        if( !stabilization[ rangeSpace.agglomeration().index( entity ) ] )
        {
          const auto &stabMatrix = rangeSpace.stabilization( entity );
          for( std::size_t r = 0; r < stabMatrix.rows(); ++r )
            for( std::size_t c = 0; c < stabMatrix.cols(); ++c )
              jLocal.add( r, c, 1*stabMatrix[ r ][ c ] );
          stabilization[ rangeSpace.agglomeration().index( entity ) ] = true;
        }

        const DomainBasisFunctionSetType &domainBaseSet = jLocal.domainBasisFunctionSet();
        const RangeBasisFunctionSetType &rangeBaseSet  = jLocal.rangeBasisFunctionSet();
        const unsigned int domainNumBasisFunctions = domainBaseSet.size();

        QuadratureType quadrature( entity, domainSpace.order()+rangeSpace.order() );
        const std::size_t numQuadraturePoints = quadrature.nop();
        for( std::size_t pt = 0; pt < numQuadraturePoints; ++pt )
        {
          const typename QuadratureType::CoordinateType &x = quadrature.point( pt );
          const double weight = quadrature.weight( pt ) * geometry.integrationElement( x );

          domainBaseSet.evaluateAll( quadrature[ pt ], phi );
          rangeBaseSet.evaluateAll( quadrature[ pt ], rphi );

          domainBaseSet.jacobianAll( quadrature[ pt ], dphi );
          rangeBaseSet.jacobianAll( quadrature[ pt ], rdphi );

          DomainRangeType u0;
          DomainJacobianRangeType jacU0;
          uLocal.evaluate( quadrature[ pt ], u0 );
          uLocal.jacobian( quadrature[ pt ], jacU0 );

          RangeRangeType aphi( 0 );
          RangeJacobianRangeType adphi( 0 );
          for( unsigned int localCol = 0; localCol < domainNumBasisFunctions; ++localCol )
          {
            model().linSource( u0, jacU0, entity, quadrature[ pt ], phi[ localCol ], dphi[ localCol ], aphi );
            model().linDiffusiveFlux( u0, jacU0, entity, quadrature[ pt ], phi[ localCol ], dphi[ localCol ], adphi );
            jLocal.column( localCol ).axpy( rphi, rdphi, aphi, adphi, weight );
          }
        }

        if( model().hasNeumanBoundary() && entity.hasBoundaryIntersections() )
        {
          for( const auto &intersection : Dune::intersections( static_cast< typename GridPartType::GridViewType >( gridPart ), entity ) )
          {
            if( !intersection.boundary() )
              continue;

            Dune::FieldVector< bool, RangeRangeType::dimension > components( true );
            bool hasDirichletComponent = model().isDirichletIntersection( intersection, components );

            const auto intersectionGeometry = intersection.geometry();
            FaceQuadratureType quadInside( gridPart, intersection, domainSpace.order()+rangeSpace.order(), FaceQuadratureType::INSIDE );
            const std::size_t numQuadraturePoints = quadInside.nop();
            for( std::size_t pt = 0; pt < numQuadraturePoints; ++pt )
            {
              const typename FaceQuadratureType::LocalCoordinateType &x = quadInside.localPoint( pt );
              double weight = quadInside.weight( pt ) * intersectionGeometry.integrationElement( x );
              DomainRangeType u0;
              uLocal.evaluate( quadInside[ pt ], u0 );
              domainBaseSet.evaluateAll( quadInside[ pt ], phi );
              for( unsigned int localCol = 0; localCol < domainNumBasisFunctions; ++localCol )
              {
                RangeRangeType alpha( 0 );
                model().linAlpha( u0, entity, quadInside[ pt ], phi[ localCol ], alpha );
                for( int k = 0; k < RangeRangeType::dimension; ++k )
                  if( hasDirichletComponent && components[ k ] )
                    alpha[ k ] = 0;
                jLocal.column( localCol ).axpy( phi, alpha, weight );
              }
            }
          }
        }
      }

      constraints().applyToOperator( jOp );

      jOp.communicate();
    }

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_OPERATOR_ELLIPTIC_HH
