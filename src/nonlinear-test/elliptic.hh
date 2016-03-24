#ifndef ELLIPTIC_HH
#define ELLIPTIC_HH

#include <dune/common/fmatrix.hh>

#include <dune/fem/quadrature/cachingquadrature.hh>
#include <dune/fem/operator/common/operator.hh>
#include <dune/fem/operator/common/stencil.hh>

#include <dune/fem/operator/common/differentiableoperator.hh>

#include "dirichletconstraints.hh"

// EllipticOperator
// ----------------

//! [Class for elliptic operator]
struct NoConstraints
{
  template <class ModelType, class DiscreteFunctionSpaceType>
  NoConstraints( const ModelType&, const DiscreteFunctionSpaceType& )
  {}
  template < class DiscreteFunctionType >
  void operator ()( const DiscreteFunctionType& u, DiscreteFunctionType& w ) const
  {}
  template < class GridFunctionType, class DiscreteFunctionType >
  void operator ()( const GridFunctionType& u, DiscreteFunctionType& w ) const
  {}
  template <class LinearOperator>
  void applyToOperator( LinearOperator& linearOperator ) const
  {}
};

template< class DomainDiscreteFunction, class RangeDiscreteFunction, class Model,
          class Constraints = Dune::DirichletConstraints< Model, typename RangeDiscreteFunction::DiscreteFunctionSpaceType > >
struct EllipticOperator
: public virtual Dune::Fem::Operator< DomainDiscreteFunction, RangeDiscreteFunction >
//! [Class for elliptic operator]
{
protected:
  typedef DomainDiscreteFunction DomainDiscreteFunctionType;
  typedef RangeDiscreteFunction  RangeDiscreteFunctionType;
  typedef Model                  ModelType;
  typedef Constraints      ConstraintsType;           // the class taking care of boundary constraints e.g. dirichlet bc

  typedef typename DomainDiscreteFunctionType::DiscreteFunctionSpaceType DomainDiscreteFunctionSpaceType;
  typedef typename DomainDiscreteFunctionType::LocalFunctionType         DomainLocalFunctionType;
  typedef typename DomainLocalFunctionType::RangeType                    DomainRangeType;
  typedef typename DomainLocalFunctionType::JacobianRangeType            DomainJacobianRangeType;
  typedef typename RangeDiscreteFunctionType::DiscreteFunctionSpaceType RangeDiscreteFunctionSpaceType;
  typedef typename RangeDiscreteFunctionType::LocalFunctionType         RangeLocalFunctionType;
  typedef typename RangeLocalFunctionType::RangeType                    RangeRangeType;
  typedef typename RangeLocalFunctionType::JacobianRangeType            RangeJacobianRangeType;

  // the following types must be identical for domain and range
  typedef typename RangeDiscreteFunctionSpaceType::IteratorType IteratorType;
  typedef typename IteratorType::Entity       EntityType;
  typedef typename EntityType::Geometry       GeometryType;
  typedef typename RangeDiscreteFunctionSpaceType::DomainType DomainType;
  typedef typename RangeDiscreteFunctionSpaceType::GridPartType  GridPartType;
  typedef typename GridPartType::IntersectionIteratorType IntersectionIteratorType;
  typedef typename IntersectionIteratorType::Intersection IntersectionType;

  typedef Dune::Fem::CachingQuadrature< GridPartType, 0 > QuadratureType;
  typedef Dune::Fem::ElementQuadrature< GridPartType, 1 > FaceQuadratureType;

public:
  //! contructor
  EllipticOperator ( const ModelType &model,
                     const RangeDiscreteFunctionSpaceType &rangeSpace )
  : model_( model )
  , constraints_( model, rangeSpace )
  {}

  // prepare the solution vector
  template <class Function>
  void prepare( const Function &func, RangeDiscreteFunctionType &u )
  {
    // set boundary values for solution
    constraints()( func, u );
  }

  //! application operator
  virtual void
  operator() ( const DomainDiscreteFunctionType &u, RangeDiscreteFunctionType &w ) const;

protected:
  const ModelType &model () const { return model_; }
  const ConstraintsType &constraints () const { return constraints_; }

private:
  ModelType model_;
  ConstraintsType constraints_;
};

// DifferentiableEllipticOperator
// ------------------------------
//! [Class for linearizable elliptic operator]
template< class JacobianOperator, class Model,
          class Constraints = Dune::DirichletConstraints< Model, typename JacobianOperator::RangeFunctionType::DiscreteFunctionSpaceType > >
struct DifferentiableEllipticOperator
: public EllipticOperator< typename JacobianOperator::DomainFunctionType, typename JacobianOperator::RangeFunctionType, Model, Constraints >,
  public Dune::Fem::DifferentiableOperator< JacobianOperator >
//! [Class for linearizable elliptic operator]
{
  typedef EllipticOperator< typename JacobianOperator::DomainFunctionType, typename JacobianOperator::RangeFunctionType, Model, Constraints > BaseType;

  typedef JacobianOperator JacobianOperatorType;

  typedef typename BaseType::DomainDiscreteFunctionType DomainDiscreteFunctionType;
  typedef typename BaseType::RangeDiscreteFunctionType  RangeDiscreteFunctionType;
  typedef typename BaseType::ModelType ModelType;

protected:
  typedef typename DomainDiscreteFunctionType::DiscreteFunctionSpaceType DomainDiscreteFunctionSpaceType;
  typedef typename DomainDiscreteFunctionType::LocalFunctionType         DomainLocalFunctionType;
  typedef typename DomainLocalFunctionType::RangeType                    DomainRangeType;
  typedef typename DomainLocalFunctionType::JacobianRangeType            DomainJacobianRangeType;
  typedef typename RangeDiscreteFunctionType::DiscreteFunctionSpaceType RangeDiscreteFunctionSpaceType;
  typedef typename RangeDiscreteFunctionType::LocalFunctionType         RangeLocalFunctionType;
  typedef typename RangeLocalFunctionType::RangeType                    RangeRangeType;
  typedef typename RangeLocalFunctionType::JacobianRangeType            RangeJacobianRangeType;

  // the following types must be identical for domain and range
  typedef typename RangeDiscreteFunctionSpaceType::IteratorType IteratorType;
  typedef typename IteratorType::Entity       EntityType;
  typedef typename EntityType::Geometry       GeometryType;
  typedef typename RangeDiscreteFunctionSpaceType::DomainType DomainType;
  typedef typename RangeDiscreteFunctionSpaceType::GridPartType  GridPartType;
  typedef typename GridPartType::IntersectionIteratorType IntersectionIteratorType;
  typedef typename IntersectionIteratorType::Intersection IntersectionType;

  typedef typename BaseType::QuadratureType QuadratureType;
  // quadrature for faces - used for Neuman b.c.
  typedef typename BaseType::FaceQuadratureType FaceQuadratureType;

public:
  //! contructor
  DifferentiableEllipticOperator ( const ModelType &model, const RangeDiscreteFunctionSpaceType &space )
  : BaseType( model, space )
  {}

  //! method to setup the jacobian of the operator for storage in a matrix
  void jacobian ( const DomainDiscreteFunctionType &u, JacobianOperatorType &jOp ) const;

protected:
  using BaseType::model;
  using BaseType::constraints;
};

// Implementation of EllipticOperator
// ----------------------------------

template< class DomainDiscreteFunction, class RangeDiscreteFunction, class Model, class Constraints >
void EllipticOperator< DomainDiscreteFunction, RangeDiscreteFunction, Model, Constraints >
  ::operator() ( const DomainDiscreteFunctionType &u, RangeDiscreteFunctionType &w ) const
{
  w.clear();
  // get discrete function space
  const RangeDiscreteFunctionSpaceType &dfSpace = w.space();

  // iterate over grid
  const IteratorType end = dfSpace.end();
  for( IteratorType it = dfSpace.begin(); it != end; ++it )
  {
    // get entity (here element)
    const EntityType &entity = *it;
    // get elements geometry
    const GeometryType &geometry = entity.geometry();

    // get local representation of the discrete functions
    const DomainLocalFunctionType uLocal = u.localFunction( entity );
    RangeLocalFunctionType wLocal = w.localFunction( entity );

    // obtain quadrature order
    const int quadOrder = uLocal.order() + wLocal.order();

    { // element integral
      QuadratureType quadrature( entity, quadOrder );
      const size_t numQuadraturePoints = quadrature.nop();
      for( size_t pt = 0; pt < numQuadraturePoints; ++pt )
      {
        //! [Compute local contribution of operator]
        const typename QuadratureType::CoordinateType &x = quadrature.point( pt );
        const double weight = quadrature.weight( pt ) * geometry.integrationElement( x );

        DomainRangeType vu;
        uLocal.evaluate( quadrature[ pt ], vu );
        DomainJacobianRangeType du;
        uLocal.jacobian( quadrature[ pt ], du );

        // compute mass contribution (studying linear case so linearizing around zero)
        RangeRangeType avu( 0 );
        model().source( entity, quadrature[ pt ], vu, du, avu );
        avu *= weight;
        // add to local functional wLocal.axpy( quadrature[ pt ], avu );

        RangeJacobianRangeType adu( 0 );
        // apply diffusive flux
        model().diffusiveFlux( entity, quadrature[ pt ], vu, du, adu );
        adu *= weight;

        // add to local function
        wLocal.axpy( quadrature[ pt ], avu, adu );
        //! [Compute local contribution of operator]
      }
    }
    if (model().hasNeumanBoundary())
    {
      if ( !entity.hasBoundaryIntersections() )
        continue;

      const IntersectionIteratorType iitend = dfSpace.gridPart().iend( entity );
      for( IntersectionIteratorType iit = dfSpace.gridPart().ibegin( entity ); iit != iitend; ++iit ) // looping over intersections
      {
        const IntersectionType &intersection = *iit;
        if ( ! intersection.boundary() )
          continue;
        Dune::FieldVector<bool,RangeRangeType::dimension> components(true);
        bool hasDirichletComponent = model().isDirichletIntersection( intersection, components);

        const typename IntersectionType::Geometry &intersectionGeometry = intersection.geometry();
        FaceQuadratureType quadInside( dfSpace.gridPart(), intersection, quadOrder, FaceQuadratureType::INSIDE );
        const size_t numQuadraturePoints = quadInside.nop();
        for( size_t pt = 0; pt < numQuadraturePoints; ++pt )
        {
          const typename FaceQuadratureType::LocalCoordinateType &x = quadInside.localPoint( pt );
          double weight = quadInside.weight( pt ) * intersectionGeometry.integrationElement( x );
          DomainRangeType vu;
          uLocal.evaluate( quadInside[ pt ], vu );
          RangeRangeType alpha( 0 );
          model().alpha( entity, quadInside[ pt ], vu, alpha );
          alpha *= weight;
          for(int k = 0; k < RangeRangeType::dimension; ++k)
            if ( hasDirichletComponent && components[k] )
              alpha[k] = 0;
          wLocal.axpy( quadInside[ pt ], alpha );
        }
      }
    }
  }

  w.communicate();
  // apply constraints, e.g. Dirichlet contraints, to the result
  constraints()( u, w );
}

// Implementation of DifferentiableEllipticOperator
// ------------------------------------------------

template< class JacobianOperator, class Model, class Constraints >
void DifferentiableEllipticOperator< JacobianOperator, Model, Constraints >
  ::jacobian ( const DomainDiscreteFunctionType &u, JacobianOperator &jOp ) const
{
  typedef typename JacobianOperator::LocalMatrixType LocalMatrixType;
  typedef typename DomainDiscreteFunctionSpaceType::BasisFunctionSetType DomainBasisFunctionSetType;
  typedef typename RangeDiscreteFunctionSpaceType::BasisFunctionSetType  RangeBasisFunctionSetType;

  const DomainDiscreteFunctionSpaceType &domainSpace = jOp.domainSpace();
  const RangeDiscreteFunctionSpaceType  &rangeSpace = jOp.rangeSpace();

  Dune::Fem::DiagonalStencil<DomainDiscreteFunctionSpaceType,RangeDiscreteFunctionSpaceType>
    stencil( domainSpace, rangeSpace );
  jOp.reserve(stencil);
  jOp.clear();

  const int domainBlockSize = domainSpace.localBlockSize; // is equal to 1 for scalar functions
  std::vector< typename DomainLocalFunctionType::RangeType >         phi( domainSpace.blockMapper().maxNumDofs()*domainBlockSize );
  std::vector< typename DomainLocalFunctionType::JacobianRangeType > dphi( domainSpace.blockMapper().maxNumDofs()*domainBlockSize );
  const int rangeBlockSize = rangeSpace.localBlockSize; // is equal to 1 for scalar functions
  std::vector< typename RangeLocalFunctionType::RangeType >         rphi( rangeSpace.blockMapper().maxNumDofs()*rangeBlockSize );
  std::vector< typename RangeLocalFunctionType::JacobianRangeType > rdphi( rangeSpace.blockMapper().maxNumDofs()*rangeBlockSize );

  const IteratorType end = rangeSpace.end();
  for( IteratorType it = rangeSpace.begin(); it != end; ++it )
  {
    const EntityType &entity = *it;
    const GeometryType &geometry = entity.geometry();

    const DomainLocalFunctionType uLocal = u.localFunction( entity );
    LocalMatrixType jLocal = jOp.localMatrix( entity, entity );

    const DomainBasisFunctionSetType &domainBaseSet = jLocal.domainBasisFunctionSet();
    const RangeBasisFunctionSetType &rangeBaseSet  = jLocal.rangeBasisFunctionSet();
    const unsigned int domainNumBasisFunctions = domainBaseSet.size();

    QuadratureType quadrature( entity, domainSpace.order()+rangeSpace.order() );
    const size_t numQuadraturePoints = quadrature.nop();
    for( size_t pt = 0; pt < numQuadraturePoints; ++pt )
    {
      //! [Assembling the local matrix]
      const typename QuadratureType::CoordinateType &x = quadrature.point( pt );
      const double weight = quadrature.weight( pt ) * geometry.integrationElement( x );

      // evaluate all basis functions at given quadrature point
      domainBaseSet.evaluateAll( quadrature[ pt ], phi );
      rangeBaseSet.evaluateAll( quadrature[ pt ], rphi );

      // evaluate jacobians of all basis functions at given quadrature point
      domainBaseSet.jacobianAll( quadrature[ pt ], dphi );
      rangeBaseSet.jacobianAll( quadrature[ pt ], rdphi );

      // get value for linearization
      DomainRangeType u0;
      DomainJacobianRangeType jacU0;
      uLocal.evaluate( quadrature[ pt ], u0 );
      uLocal.jacobian( quadrature[ pt ], jacU0 );

      RangeRangeType aphi( 0 );
      RangeJacobianRangeType adphi( 0 );
      for( unsigned int localCol = 0; localCol < domainNumBasisFunctions; ++localCol )
      {
        // if mass terms or right hand side is present
        model().linSource( u0, jacU0, entity, quadrature[ pt ], phi[ localCol ], dphi[localCol], aphi );

        // if gradient term is present
        model().linDiffusiveFlux( u0, jacU0, entity, quadrature[ pt ], phi[ localCol ], dphi[ localCol ], adphi );

        // get column object and call axpy method
        jLocal.column( localCol ).axpy( rphi, rdphi, aphi, adphi, weight );
      }
      //! [Assembling the local matrix]
    }

    if (model().hasNeumanBoundary())
    {
      if ( !entity.hasBoundaryIntersections() )
        continue;

      const IntersectionIteratorType iitend = rangeSpace.gridPart().iend( entity );
      for( IntersectionIteratorType iit = rangeSpace.gridPart().ibegin( entity ); iit != iitend; ++iit ) // looping over intersections
      {
        const IntersectionType &intersection = *iit;
        if ( ! intersection.boundary() )
          continue;
        Dune::FieldVector<bool,RangeRangeType::dimension> components(true);
        bool hasDirichletComponent = model().isDirichletIntersection( intersection, components);

        const typename IntersectionType::Geometry &intersectionGeometry = intersection.geometry();
        FaceQuadratureType quadInside( rangeSpace.gridPart(), intersection, domainSpace.order()+rangeSpace.order(), FaceQuadratureType::INSIDE );
        const size_t numQuadraturePoints = quadInside.nop();
        for( size_t pt = 0; pt < numQuadraturePoints; ++pt )
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
            for(int k = 0; k < RangeRangeType::dimension; ++k)
              if ( hasDirichletComponent && components[k] )
                alpha[k] = 0;
            jLocal.column( localCol ).axpy( phi, alpha, weight );
          }
        }
      }
    }
  } // end grid traversal

  // apply constraints to matrix operator
  constraints().applyToOperator( jOp );
  jOp.communicate();
}

#endif // #ifndef ELLIPTIC_HH
