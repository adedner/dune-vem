#ifndef SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_CURVEDRIDGES_HH
#define SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_CURVEDRIDGES_HH

#include <cassert>
#include <cmath>

#include "interface.hh"


// CurvedRidges
// ------------

template< class FunctionSpace >
class CurvedRidges
  : public ProblemInterface< FunctionSpace >
{
  typedef ProblemInterface< FunctionSpace >  BaseType;

public:
  typedef typename BaseType::RangeType RangeType;
  typedef typename BaseType::DomainType DomainType;
  typedef typename BaseType::JacobianRangeType JacobianRangeType;
  typedef typename BaseType::DiffusionTensorType DiffusionTensorType;

  enum { dimRange  = BaseType::dimRange };
  enum { dimDomain = BaseType::dimDomain };

  //! the right hand side data (default = 0)
  virtual void f ( const DomainType &p,
                   RangeType &phi ) const
  {
    double q = p[ 0 ];
    for( int i = 1; i < dimDomain; ++i )
      q += std::sin( 10*p[ i ]+5*p[ 0 ]*p[ 0 ] );
    const double u = std::exp( q );
    double t1 = 1, t2 = 0, t3 = 0;
    for( int i = 1; i < dimDomain; ++i )
    {
      t1 += std::cos( 10*p[ i ]+5*p[ 0 ]*p[ 0 ] ) * 10 * p[ 0 ];
      t2 += 10*std::cos( 10*p[ i ]+5*p[ 0 ]*p[ 0 ] ) -
            100*std::sin( 10*p[ i ]+5*p[ 0 ]*p[ 0 ] ) * p[ 0 ]*p[ 0 ];
      t3 += 100*std::cos( 10*p[ i ]+5*p[ 0 ]*p[ 0 ] )*std::cos( 10*p[ i ]+5*p[ 0 ]*p[ 0 ] ) -
            100*std::sin( 10*p[ i ]+5*p[ 0 ]*p[ 0 ] );
    }
    t1 = t1*t1;

    phi = -u*( t1+t2+t3 );
  }

  //! the exact solution
  virtual void u ( const DomainType &p,
                   RangeType &phi ) const
  {
    double q = p[ 0 ];
    for( int i = 1; i < dimDomain; ++i )
      q += std::sin( 10*p[ i ]+5*p[ 0 ]*p[ 0 ] );
    const double exponential = std::exp( q );
    phi = exponential;
  }

  //! the jacobian of the exact solution
  virtual void uJacobian ( const DomainType &x,
                           JacobianRangeType &value ) const
  {
    DUNE_THROW( Dune::NotImplemented, "still todo" );
  }

  //! return true if Dirichlet boundary is present (default is true)
  virtual bool hasDirichletBoundary () const
  {
    return true;
  }

  //! return true if given point belongs to the Dirichlet boundary (default is true)
  virtual bool isDirichletPoint ( const DomainType &x ) const
  {
    return true;
  }
};

#endif // #ifndef SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_CURVEDRIDGES_HH
