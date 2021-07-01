#ifndef SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_COSINUSPRODUCTMIXEDBC_HH
#define SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_COSINUSPRODUCTMIXEDBC_HH

#include <cassert>
#include <cmath>

#include "interface.hh"


// CosinusProductMixedBC
// ---------------------

template< class FunctionSpace >
class CosinusProductMixedBC
  : public ProblemInterface< FunctionSpace >
{
  typedef ProblemInterface< FunctionSpace >  BaseType;

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
      phi *= std::cos( 2*M_PI*x[ i ] + 1. );
  }

  //! the exact solution
  virtual void u ( const DomainType &x, RangeType &phi ) const override
  {
    phi = 1;
    for( int i = 0; i < dimDomain; ++i )
      phi *= std::cos( 2*M_PI*x[ i ] + 1. );
  }

  //! the jacobian of the exact solution
  virtual void uJacobian ( const DomainType &x, JacobianRangeType &ret ) const override
  {
    for( int r = 0; r < dimRange; ++r )
      for( int i = 0; i < dimDomain; ++i )
      {
        ret[ r ][ i ] = -2*M_PI*std::sin( 2*M_PI*x[ i ] +1. );
        for( int j = 1; j < dimDomain; ++j )
          ret[ r ][ i ] *= std::cos( 2*M_PI*x[ ( i+j )%dimDomain ] +1. );
      }
  }

  //! mass coefficient can be 0 for this problem
  virtual void m ( const DomainType &x, RangeType &m ) const override { m = RangeType( 0 ); }

  virtual void alpha ( const DomainType &x, RangeType &a ) const override
  {
    a = RangeType( 0.5 );
  }

  //! the Dirichlet boundary data (default calls u)
  virtual void g ( const DomainType &x, RangeType &value ) const override { u( x, value ); }

  virtual bool hasDirichletBoundary () const override { return true; }
  virtual bool hasNeumanBoundary () const override { return true; }

  virtual bool isDirichletPoint ( const DomainType &x ) const override
  {
    // all boundaries except the x=0 plane are Dirichlet
    return (std::abs( x[ 0 ] ) > 1e-8);
  }

  virtual void n ( const DomainType &x, RangeType &value ) const override
  {
    u( x, value );
    value *= 0.5;
    JacobianRangeType jac;
    uJacobian( x, jac );
    value[ 0 ] -= jac[ 0 ][ 0 ];
  }
};

#endif // #ifndef SRC_POISSON_TEST_AGGLOMERATION_DG_PROBLEMS_COSINUSPRODUCTMIXEDBC_HH
