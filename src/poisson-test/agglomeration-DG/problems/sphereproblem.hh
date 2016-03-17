#ifndef SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_SPHEREPROBLEM_HH
#define SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_SPHEREPROBLEM_HH

#include <cassert>
#include <cmath>

#include "interface.hh"


// SphereProblem
// -------------

// A problem on a unit sphere
template< class FunctionSpace >
class SphereProblem : public ProblemInterface< FunctionSpace >
{
  typedef ProblemInterface< FunctionSpace >  BaseType;

public:
  typedef typename BaseType::RangeType RangeType;
  typedef typename BaseType::DomainType DomainType;
  typedef typename BaseType::JacobianRangeType JacobianRangeType;
  typedef typename BaseType::DiffusionTensorType DiffusionTensorType;

  typedef typename FunctionSpace::HessianRangeType HessianRangeType;

  enum { dimRange  = BaseType::dimRange };
  enum { dimDomain = BaseType::dimDomain };

  //! the right hand side data (default = 0)
  virtual void f ( const DomainType &x,
                   RangeType &ret ) const
  {
    DomainType y = x;
    y /= x.two_norm();

    JacobianRangeType Ju;
    uJacobian( y, Ju );
    Ju.mv( y, ret );
    ret *= double(dimDomain - 1);

#if 0
    HessianRangeType Hu;
    hessian( y, Hu );
    for( int i = 0; i < dimDomain; ++i )
    {
      DomainType ds = y;
      ds *= -y[ i ];
      ds[ i ] += 1;

      for( int j = 0; j < dimRange; ++j )
      {
        DomainType Hds;
        Hu[ j ].mv( ds, Hds );
        ret[ j ] -= ds * Hds;
      }
    }
#endif
    RangeType uVal;
    u( y, uVal );
    ret += uVal;
  }

  //! exact solution
  virtual void u ( const DomainType &x,
                   RangeType &phi ) const
  {
    assert( dimRange == 1 );
    phi[ 0 ] = 1;
    for( int i = 0; i < dimDomain; ++i )
      phi[ 0 ] *= std::cos( 2*M_PI*x[ i ] );
  }

  //! the jacobian of the exact solution
  virtual void uJacobian ( const DomainType &x,
                           JacobianRangeType &ret ) const
  {
    assert( dimRange == 1 );
    for( int i = 0; i < dimDomain; ++i )
    {
      ret[ 0 ][ i ] = -2*M_PI*std::sin( 2*M_PI*x[ i ] );
      for( int j = 1; j < dimDomain; ++j )
        ret[ 0 ][ i ] *= std::cos( 2*M_PI*x[ ( i+j )%dimDomain ] );
    }
  }

  //! mass coefficient has to be 1 for this problem
  virtual void m ( const DomainType &x, RangeType &m ) const
  {
    m = RangeType( 1 );
  }

private:
  void hessian ( const DomainType &x, HessianRangeType &ret ) const
  {
    assert( dimRange == 1 );
    for( int i = 0; i < dimDomain; ++i )
      for( int j = 0; j < dimDomain; ++j )
      {
        if( j == i )
        {
          ret[ 0 ][ i ][ j ] = -4*M_PI*M_PI;
          for( int k = 0; k < dimDomain; ++k )
            ret[ 0 ][ i ][ j ] *= std::cos( 2*M_PI*x[ k ] );
        }
        else
        {
          ret[ 0 ][ i ][ j ] = 4*M_PI*M_PI;
          for( int k = 0; k < dimDomain; ++k )
            if( k != i && k != j )
              ret[ 0 ][ i ][ j ] *= std::cos( 2*M_PI*x[ k ] );
            else
              ret[ 0 ][ i ][ j ] *= std::sin( 2*M_PI*x[ k ] );
        }
      }
  }
};

#endif // #ifndef SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_SPHEREPROBLEM_HH
