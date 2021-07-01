#ifndef SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_REENTRANTCORNER_HH
#define SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_REENTRANTCORNER_HH

#include <cassert>
#include <cmath>

#include "interface.hh"


// ReentrantCorner
// ---------------

// a reentrant corner problem with a 270 degree corner
template< class FunctionSpace >
class ReentrantCorner : public ProblemInterface< FunctionSpace >
{
  typedef ProblemInterface< FunctionSpace >  BaseType;

  const double lambda_;

public:
  typedef typename BaseType::RangeType RangeType;
  typedef typename BaseType::DomainType DomainType;
  typedef typename BaseType::JacobianRangeType JacobianRangeType;
  typedef typename BaseType::DiffusionTensorType DiffusionTensorType;

  enum { dimRange  = BaseType::dimRange };
  enum { dimDomain = BaseType::dimDomain };

  ReentrantCorner () : lambda_( 180./270. ) {}

  //! the right hand side data (default = 0)
  virtual void f ( const DomainType &x,
                   RangeType &phi ) const
  {
    phi = 0;
  }

  //! the exact solution
  virtual void u ( const DomainType &x,
                   RangeType &ret ) const
  {
    double r2 = x.two_norm2();
    double phi = argphi( x[ 0 ], x[ 1 ] );
    ret = std::pow( r2, lambda_ * 0.5 ) * std::sin( lambda_*phi );
  }

  //! the jacobian of the exact solution
  virtual void uJacobian ( const DomainType &x,
                           JacobianRangeType &ret ) const
  {
    double grad[ 2 ] = { 0.0, 0.0 };

    double r2 = x.two_norm2();
    double phi = argphi( x[ 0 ], x[ 1 ] );
    double r2dx = 2.*x[ 0 ];
    double r2dy = 2.*x[ 1 ];
    double phidx = -x[ 1 ]/r2;
    double phidy = x[ 0 ]/r2;
    double lambdaPow = ( lambda_*0.5 )*pow( r2, lambda_*0.5-1. );
    grad[ 0 ] = lambdaPow * r2dx * sin( lambda_ * phi )
                + lambda_ * cos( lambda_ * phi ) * phidx * pow( r2, lambda_ * 0.5 );
    grad[ 1 ] = lambdaPow * r2dy * sin( lambda_ * phi )
                + lambda_ * cos( lambda_ * phi ) * phidy * pow( r2, lambda_*0.5 );

    assert( grad[ 0 ] == grad[ 0 ] );
    assert( grad[ 1 ] == grad[ 1 ] );

    assert( dimRange == 1 );
    for( int i = 0; i < dimDomain; ++i )
      ret[ 0 ][ i ] = grad[ i ];
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

protected:
  /** \brief proper implementation of atan(x,y)
   */
  inline double argphi ( double x, double y ) const
  {
    double phi = std::arg( std::complex< double >( x, y ));
    if( y < 0 )
      phi += 2.*M_PI;
    return phi;
  }
};

#endif // #ifndef SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_REENTRANTCORNER_HH
