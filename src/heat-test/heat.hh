#ifndef POISSON_PROBLEMS_HH
#define POISSON_PROBLEMS_HH

#include <cassert>
#include <cmath>

#include "temporalprobleminterface.hh"

template <class FunctionSpace>
class TimeDependentCosinusProduct : public TemporalProblemInterface < FunctionSpace >
{
  typedef TemporalProblemInterface < FunctionSpace >  BaseType;
public:
  typedef typename BaseType :: RangeType            RangeType;
  typedef typename BaseType :: DomainType           DomainType;
  typedef typename BaseType :: JacobianRangeType    JacobianRangeType;
  typedef typename BaseType :: DiffusionTensorType  DiffusionTensorType;

  enum { dimRange  = BaseType :: dimRange };
  enum { dimDomain = BaseType :: dimDomain };

  // get time function from base class
  using BaseType :: time ;

  TimeDependentCosinusProduct( const Dune::Fem::TimeProviderBase &timeProvider )
    : BaseType( timeProvider )
  {}

  //! the right hand side data (default = 0)
  virtual void f(const DomainType& x,
                 RangeType& phi) const
  {
    phi  = M_PI*(4*dimDomain*M_PI*std::cos( M_PI*time() ) - std::sin( M_PI*time() ));
    for( int i = 0; i < dimDomain; ++i )
      phi *= std::cos( 2*M_PI*x[ i ] );
  }

  //! the exact solution
  virtual void u(const DomainType& x,
                 RangeType& phi) const
  {
    phi = std::cos( M_PI*time() );
    for( int i = 0; i < dimDomain; ++i )
      phi *= std::cos( 2*M_PI*x[ i ] );
  }

  //! the jacobian of the exact solution
  virtual void uJacobian(const DomainType& x,
                         JacobianRangeType& ret) const
  {
    for( int r = 0; r < dimRange; ++ r )
    {
      for( int i = 0; i < dimDomain; ++i )
      {
        ret[ r ][ i ] = -2*M_PI*std::cos( M_PI*time() )*std::sin( 2*M_PI*x[ i ] );
        for( int j = 1; j < dimDomain; ++j )
          ret[ r ][ i ] *= std::cos( 2*M_PI*x[ (i+j)%dimDomain ] );
      }
    }
  }
};

#endif // #ifndef POISSON_HH
