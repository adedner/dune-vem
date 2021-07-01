#ifndef SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_ANISOTROPIC_HH
#define SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_ANISOTROPIC_HH

#include <cassert>
#include <cmath>

#include "interface.hh"


// CosinusProduct
// --------------

// -laplace u + u = f with Neumann boundary conditions on domain [0,1]^d
// Exsct solution is u(x_1,...,x_d) = cos(2*pi*x_1)...cos(2*pi*x_d)
template< class FunctionSpace >
class AnisotropicProblem
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
    RangeType uValue;
    u( x, uValue );

    JacobianRangeType uJac;
    uJacobian( x, uJac );

    RangeType gamma;
    m( x, gamma );

    phi = M_PI*M_PI * uValue[ 0 ] * ( 2.0 + x[ 0 ]*x[ 0 ]+x[ 1 ]*x[ 1 ] );
    phi += x[ 1 ] * uValue[ 0 ] * uJac[ 0 ][ 1 ];
    phi += x[ 0 ] * uValue[ 0 ] * uJac[ 0 ][ 0 ];
    phi += 4.0 * M_PI*M_PI * x[ 0 ] * x[ 1 ] * uValue[ 0 ] * std::cos( M_PI * x[ 0 ] ) * std::cos( M_PI * x[ 1 ] );
    phi += gamma * uValue[ 0 ];
  }

  //! diffusion coefficient (default = Id)
  virtual DiffusionTensorType diffusionTensor ( const DomainType &x ) const override
  {
    DiffusionTensorType D;
    D[ 0 ][ 0 ] = 1 + x[ 1 ]*x[ 1 ];
    D[ 1 ][ 1 ] = 1 + x[ 0 ]*x[ 0 ];
    D[ 1 ][ 0 ] = D[ 0 ][ 1 ] = -x[ 0 ]*x[ 1 ]*std::sin( M_PI*x[ 0 ] )*std::sin( M_PI*x[ 1 ] );
    return D;
  }

  //! the exact solution
  virtual void u ( const DomainType &x, RangeType &phi ) const override
  {
    phi = 1;
    for( int i = 0; i < dimDomain; ++i )
      phi *= std::sin( M_PI*x[ i ] );
  }

  //! the jacobian of the exact solution
  virtual void uJacobian ( const DomainType &x, JacobianRangeType &ret ) const override
  {
    for( int r = 0; r < dimRange; ++r )
      for( int i = 0; i < dimDomain; ++i )
      {
        ret[ r ][ i ] = M_PI*std::cos( M_PI*x[ i ] );
        for( int j = 1; j < dimDomain; ++j )
          ret[ r ][ i ] *= std::sin( M_PI*x[ ( i+j )%dimDomain ] );
      }
  }

  virtual void m ( const DomainType &x, RangeType &m ) const override { m = (1.0 + x[ 0 ]*x[ 0 ] + x[ 1 ]*x[ 1 ]*x[ 1 ]); }

  virtual bool isDirichletPoint ( const DomainType &x ) const override { return true; }

  virtual bool hasDirichletBoundary () const override { return true; }
};

#endif // #ifndef SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_ANISOTROPIC_HH
