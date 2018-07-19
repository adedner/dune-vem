#ifndef HEAT_VEMSCHEME_HH
#define HEAT_VEMSCHEME_HH

#include <dune/fem/function/common/gridfunctionadapter.hh>
#include <dune/vem/space/interpolate.hh>

// local includes
#include "vemscheme.hh"

// HeatScheme
//-----------

template < class ImplicitModel, class ExplicitModel >
struct HeatScheme : public VemScheme<ImplicitModel>
{
  typedef VemScheme<ImplicitModel> BaseType;
  typedef typename BaseType::GridType GridType;
  typedef typename BaseType::GridPartType GridPartType;
  typedef typename BaseType::ModelType ImplicitModelType;
  typedef ExplicitModel ExplicitModelType;
  typedef typename BaseType::FunctionSpaceType FunctionSpaceType;
  typedef typename BaseType::DiscreteFunctionType DiscreteFunctionType;
  HeatScheme( GridPartType &gridPart,
              const ImplicitModelType& implicitModel,
              const ExplicitModelType& explicitModel,
              typename BaseType::AgglomerationType& agglomeration )
  : BaseType(gridPart, implicitModel, agglomeration),
    explicitModel_(explicitModel),
    explicitOperator_( explicitModel_, discreteSpace_ )
  {
  }

  void prepare()
  {
    // apply constraints, e.g. Dirichlet contraints, to the solution
    explicitOperator_.prepare( explicitModel_.dirichletBoundary(), solution_ );
    // apply explicit operator and also setup right hand side
    explicitOperator_( solution_, rhs_ );
    // apply constraints, e.g. Dirichlet contraints, to the result
    explicitOperator_.prepare( solution_, rhs_ );
  }

  void initialize ()
  {
    typedef Dune::Fem::GridFunctionAdapter< typename ExplicitModelType::InitialFunctionType, GridPartType > GridInitialFunction;
    GridInitialFunction gridInitialFunction( "initial data", explicitModel_.initialFunction(), solution_.gridPart() );
    Dune::Fem::interpolate( gridInitialFunction, solution_ );
  }

private:
  using BaseType::gridPart_;
  using BaseType::discreteSpace_;
  using BaseType::solution_;
  using BaseType::implicitModel_;
  using BaseType::rhs_;
  const ExplicitModelType &explicitModel_;
  typename BaseType::EllipticOperatorType explicitOperator_; // the operator for the rhs
};

#endif // end #if HEAT_VEMSCHEME_HH
