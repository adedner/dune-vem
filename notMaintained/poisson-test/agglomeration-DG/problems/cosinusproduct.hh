#ifndef SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_COSINUSPRODUCT_HH
#define SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_COSINUSPRODUCT_HH

#include <cassert>
#include <cmath>

#include "interface.hh"


// CosinusProduct
// --------------

// -laplace u + u = f with Neumann boundary conditions on domain [0,1]^d
// Exsct solution is u(x_1,...,x_d) = cos(2*pi*x_1)...cos(2*pi*x_d)
template< class FunctionSpace >
class CosinusProduct
  : public ProblemInterface< FunctionSpace >
{
  typedef ProblemInterface< FunctionSpace > BaseType;

public:
  typedef typename BaseType::RangeType RangeType;
  typedef typename BaseType::DomainType DomainType;
  typedef typename BaseType::JacobianRangeType JacobianRangeType;
  typedef typename BaseType::DiffusionTensorType DiffusionTensorType;

  const int dimRange = BaseType::dimRange;
  const int dimDomain = BaseType::dimDomain;

  //! the right hand side data (default = 0)
  virtual void f ( const DomainType &x, RangeType &phi ) const override
  {
    phi = 4*dimDomain*( M_PI*M_PI );
    for( int i = 0; i < dimDomain; ++i )
      phi *= std::cos( 2*M_PI*x[ i ] );
    RangeType uVal;
    u( x, uVal );
    phi += uVal;
  }

  //! the exact solution
  virtual void u ( const DomainType &x, RangeType &phi ) const override
  {
    phi = 1;
    for( int i = 0; i < dimDomain; ++i )
      phi *= std::cos( 2*M_PI*x[ i ] );
  }

  //! the jacobian of the exact solution
  virtual void uJacobian ( const DomainType &x, JacobianRangeType &ret ) const override
  {
    for( int r = 0; r < dimRange; ++r )
      for( int i = 0; i < dimDomain; ++i )
      {
        ret[ r ][ i ] = -2*M_PI*std::sin( 2*M_PI*x[ i ] );
        for( int j = 1; j < dimDomain; ++j )
          ret[ r ][ i ] *= std::cos( 2*M_PI*x[ ( i+j )%dimDomain ] );
      }
  }

  //! mass coefficient has to be 1 for this problem
  virtual void m ( const DomainType &x, RangeType &m ) const override { m = RangeType( 1 ); }
};

#endif // #ifndef SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_COSINUSPRODUCT_HH
