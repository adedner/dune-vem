#ifndef HEAT_MODEL_HH
#define HEAT_MODEL_HH

#include <cassert>
#include <cmath>

#include <dune/fem/solver/timeprovider.hh>
#include <dune/fem/io/parameter.hh>
#include <dune/fem/function/common/gridfunctionadapter.hh>

#include "temporalprobleminterface.hh"
#include "model.hh"

template< class FunctionSpace, class GridPart >
struct HeatModel : public DiffusionModel<FunctionSpace,GridPart>
{
  typedef DiffusionModel<FunctionSpace,GridPart> BaseType;
  typedef FunctionSpace FunctionSpaceType;
  typedef GridPart GridPartType;

  typedef typename FunctionSpaceType::DomainType DomainType;
  typedef typename FunctionSpaceType::RangeType RangeType;
  typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;

  typedef typename FunctionSpaceType::DomainFieldType DomainFieldType;
  typedef typename FunctionSpaceType::RangeFieldType RangeFieldType;

  typedef TemporalProblemInterface< FunctionSpaceType > ProblemType ;

  typedef typename BaseType::ProblemType InitialFunctionType;

  typedef Dune::Fem::TimeProviderBase TimeProviderType;

  static const int dimRange = FunctionSpaceType::dimRange;

  //! constructor taking problem reference, time provider,
  //! time step factor( either theta or -(1-theta) ),
  //! flag for the right hand side
  HeatModel( const ProblemType& problem,
             const GridPart &gridPart,
             const bool implicit )
    : BaseType(problem,gridPart),
      timeProvider_(problem.timeProvider()),
      implicit_( implicit ),
      timeStepFactor_( 0 )
  {
    // get theta for theta scheme
    const double theta = Dune::Fem::Parameter::getValue< double >("heat.theta", 0.5 );
    if (implicit)
      timeStepFactor_ = theta ;
    else
      timeStepFactor_ = -( 1.0 - theta ) ;
  }

  using BaseType::entity;

  template< class Point >
  void source ( const Point &x,
                const RangeType &value,
                const JacobianRangeType &gradient,
                RangeType &flux ) const
  {
    linSource( value, gradient, x, value, gradient, flux );
    // the explicit model should also evaluate the RHS
    if( !implicit_ )
    {
      const DomainType xGlobal = entity().geometry().global( coordinate( x ) );
      // evaluate right hand side
      RangeType rhs ;
      problem_.f( xGlobal, rhs );
      rhs  *= timeProvider_.deltaT();
      flux += rhs ;
    }
  }

  template< class Point >
  void linSource ( const RangeType& uBar,
                   const JacobianRangeType &gradientBar,
                   const Point &x,
                   const RangeType &value,
                   const JacobianRangeType &gradient,
                   RangeType &flux ) const
  {
    const DomainType xGlobal = entity().geometry().global( coordinate( x ) );
    RangeType m;
    problem_.m(xGlobal,m);
    for (unsigned int i=0;i<flux.size();++i)
      flux[i] = m[i]*value[i];
    flux *= timeStepFactor_ * timeProvider_.deltaT();
    // add term from time derivative
    flux += value;
  }

   //! return the diffusive flux
  template< class Point >
  void diffusiveFlux ( const Point &x,
                       const RangeType &value,
                       const JacobianRangeType &gradient,
                       JacobianRangeType &flux ) const
  {
    linDiffusiveFlux( value, gradient, x, value, gradient, flux );
  }
  //! return the diffusive flux
  template< class Point >
  void linDiffusiveFlux ( const RangeType& uBar,
                          const JacobianRangeType &gradientBar,
                          const Point &x,
                          const RangeType &value,
                          const JacobianRangeType &gradient,
                          JacobianRangeType &flux ) const
  {
    flux  = gradient;
    flux *= timeStepFactor_ * timeProvider_.deltaT();
  }
  template< class Point >
  void alpha(const Point &x,
             const RangeType &value,
             RangeType &val) const
  {
    BaseType::alpha(x,value,val);
  }
  template< class Point >
  void linAlpha(const RangeType &uBar,
                const Point &x,
                const RangeType &value,
                RangeType &val) const
  {
    BaseType::linAlpha(uBar,x,value,val);
  }
  //! exact some methods from the problem class
  bool hasDirichletBoundary () const
  {
    return BaseType::hasDirichletBoundary() ;
  }
  bool hasNeumanBoundary () const
  {
    return BaseType::hasNeumanBoundary() ;
  }

  //! return true if given point belongs to the Dirichlet boundary (default is true)
  template <class Intersection>
  bool isDirichletIntersection( const Intersection& inter, Dune::FieldVector<int,dimRange> &dirichletComponent ) const
  {
    return BaseType::isDirichletIntersection(inter,dirichletComponent) ;
  }

  template< class Entity, class Point >
  void g( const RangeType& uBar,
          const Entity &entity,
          const Point &x,
          RangeType &u ) const
  {
    BaseType::g(uBar,entity,x,u);
  }

  // return Fem::Function for Dirichlet boundary values
  typename BaseType::DirichletBoundaryType dirichletBoundary( ) const
  {
    return BaseType::dirichletBoundary();
  }

  //! return reference to Problem's time provider
  const TimeProviderType & timeProvider() const
  {
    return timeProvider_;
  }

  const InitialFunctionType &initialFunction() const
  {
    return problem_;
  }

protected:
  using BaseType::problem_;
  const TimeProviderType &timeProvider_;
  bool implicit_;
  double timeStepFactor_;

};
#endif // #ifndef HEAT_MODEL_HH
