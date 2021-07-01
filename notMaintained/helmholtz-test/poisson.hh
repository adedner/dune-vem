#ifndef POISSON_PROBLEMS_HH
#define POISSON_PROBLEMS_HH

#include <cassert>
#include <cmath>

#include "probleminterface.hh"

// -laplace u + u = f with Neumann boundary conditions on domain [0,1]^d
// Exsct solution is u(x_1,...,x_d) = cos(2*pi*x_1)...cos(2*pi*x_d)
template <class FunctionSpace>
class CosinusProduct : public ProblemInterface < FunctionSpace >
{
  typedef ProblemInterface < FunctionSpace >  BaseType;
public:
  typedef typename BaseType :: RangeType            RangeType;
  typedef typename BaseType :: DomainType           DomainType;
  typedef typename BaseType :: JacobianRangeType    JacobianRangeType;
  typedef typename BaseType :: DiffusionTensorType  DiffusionTensorType;

  enum { dimRange  = BaseType :: dimRange };
  enum { dimDomain = BaseType :: dimDomain };

  //! the right hand side data (default = 0)
  virtual void f(const DomainType& x,
                 RangeType& phi) const
  {
    phi = 4*dimDomain*(M_PI*M_PI);
    for( int i = 0; i < dimDomain; ++i )
      phi *= std::cos( 2*M_PI*x[ i ] );
    RangeType uVal;
    u(x,uVal);
    phi += uVal;
  }

  //! the exact solution
  virtual void u(const DomainType& x,
                 RangeType& phi) const
  {
    phi = 1;
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
        ret[ r ][ i ] = -2*M_PI*std::sin( 2*M_PI*x[ i ] );
        for( int j = 1; j < dimDomain; ++j )
          ret[ r ][ i ] *= std::cos( 2*M_PI*x[ (i+j)%dimDomain ] );
      }
    }
  }
  //! mass coefficient has to be 1 for this problem
  virtual void m(const DomainType& x, RangeType &m) const
  {
    m = RangeType(1);
  }
};
// -laplace u = f with zero Dirichlet boundary conditions on domain [0,1]^d
// Exsct solution is u(x_1,...,x_d) = sin(2*pi*x_1)...sin(2*pi*x_d)
template <class FunctionSpace>
class SinusProduct : public ProblemInterface < FunctionSpace >
{
  typedef ProblemInterface < FunctionSpace >  BaseType;
public:
  typedef typename BaseType :: RangeType            RangeType;
  typedef typename BaseType :: DomainType           DomainType;
  typedef typename BaseType :: JacobianRangeType    JacobianRangeType;
  typedef typename BaseType :: DiffusionTensorType  DiffusionTensorType;

  enum { dimRange  = BaseType :: dimRange };
  enum { dimDomain = BaseType :: dimDomain };

  //! the right hand side data (default = 0)
  virtual void f(const DomainType& x,
      RangeType& phi) const
  {
    phi = 0;
  }

  //! the exact solution
  virtual void u(const DomainType& x,
      RangeType& phi) const
  {

    std::complex<double> com_one(0,1.);
    phi = std::exp(com_one * M_PI * x[0]);

  }

  //! the jacobian of the exact solution
  virtual void uJacobian(const DomainType& x,
      JacobianRangeType& ret) const
  {
    std::complex<double> com_one(0,1.);
    ret[0][0] = com_one * M_PI * std::exp( com_one * M_PI * x[0]);
    ret[0][1] = 0;

  }

  //! mass coefficient has to be 1 for this problem
  virtual void m(const DomainType& x, RangeType &m) const
  {
    m = - M_PI * M_PI;
  }

  virtual void n(const DomainType& x,
      RangeType& value) const
  {
    std::complex<double> com_one(0,1.);
    // evaluate uTrue
    u(x,value);

    // multiply uTrue with (ik)
    value *= com_one * M_PI;

    // evaluate derivative of uTrue
    JacobianRangeType jac;
    uJacobian(x,jac);

    // multiply derivative of uTrue with respective normals
    if (x[1] == 0.0 )
    {
      jac [0][0] *=  0.0;
      jac [0][1] *= 1.0;
    }

    if (x[0] == 1.0)
    {
      jac [0][0] *=  1.0;
      jac [0][1] *= 0.0;
    }

    if (x[1] == 1.0)
    {
      jac [0][0] *=  0.0;
      jac [0][1] *= -1.0;
    }

    if (x[0] == 0.0)
    {
      jac [0][0] *=  -1.0;
      jac [0][1] *= 0.0;
    }

    value[0] += jac[0][0] +  jac[0][1];
  }

  //  //! return true if given point belongs to the Dirichlet boundary (default is true)
  //  virtual bool isDirichletPoint( const DomainType& x ) const
  //  {
  //    return true;
  //  }
  //  virtual bool hasDirichletBoundary () const
  //  {
  //    return true ;
  //  }
};
template <class FunctionSpace>
class CosinusProductMixedBC : public ProblemInterface < FunctionSpace >
{
  typedef ProblemInterface < FunctionSpace >  BaseType;
public:
  typedef typename BaseType :: RangeType            RangeType;
  typedef typename BaseType :: DomainType           DomainType;
  typedef typename BaseType :: JacobianRangeType    JacobianRangeType;
  typedef typename BaseType :: DiffusionTensorType  DiffusionTensorType;

  enum { dimRange  = BaseType :: dimRange };
  enum { dimDomain = BaseType :: dimDomain };

  //! the right hand side data (default = 0)
  virtual void f(const DomainType& x,
                 RangeType& phi) const
  {
    phi = 4*dimDomain*(M_PI*M_PI);
    for( int i = 0; i < dimDomain; ++i )
      phi *= std::cos( 2*M_PI*x[ i ] + 1. );
  }

  //! the exact solution
  virtual void u(const DomainType& x,
                 RangeType& phi) const
  {
    phi = 1;
    for( int i = 0; i < dimDomain; ++i )
      phi *= std::cos( 2*M_PI*x[ i ] + 1. );
  }

  //! the jacobian of the exact solution
  virtual void uJacobian(const DomainType& x,
                         JacobianRangeType& ret) const
  {
    for( int r = 0; r < dimRange; ++ r )
    {
      for( int i = 0; i < dimDomain; ++i )
      {
        ret[ r ][ i ] = -2*M_PI*std::sin( 2*M_PI*x[ i ] +1. );
        for( int j = 1; j < dimDomain; ++j )
          ret[ r ][ i ] *= std::cos( 2*M_PI*x[ (i+j)%dimDomain ] +1. );
      }
    }
  }

  //! mass coefficient can be 0 for this problem
  virtual void m(const DomainType& x, RangeType &m) const
  {
    m = RangeType(0);
  }
  virtual void alpha(const DomainType& x, RangeType &a) const
  {
    a = RangeType(0.5);
  }
  //! the Dirichlet boundary data (default calls u)
  virtual void g(const DomainType& x,
                 RangeType& value) const
  {
    u(x,value);
  }
  virtual bool hasDirichletBoundary () const
  {
    return true ;
  }
  virtual bool hasNeumanBoundary () const
  {
    return true ;
  }
  virtual bool isDirichletPoint( const DomainType& x ) const
  {
    // all boundaries except the x=0 plane are Dirichlet
    return (std::abs(x[0])>1e-8);

  }
  virtual void n(const DomainType& x,
                 RangeType& value) const
  {

  std::cout << "GCD1" << std::endl;
  u(x,value);
    value *= 0.5;
    JacobianRangeType jac;
    uJacobian(x,jac);
    value[0] -= jac[0][0];
  }
};

// A problem on a unit sphere
template <class FunctionSpace>
class SphereProblem : public ProblemInterface < FunctionSpace >
{
  typedef ProblemInterface < FunctionSpace >  BaseType;
public:
  typedef typename BaseType :: RangeType            RangeType;
  typedef typename BaseType :: DomainType           DomainType;
  typedef typename BaseType :: JacobianRangeType    JacobianRangeType;
  typedef typename BaseType :: DiffusionTensorType  DiffusionTensorType;

  typedef typename FunctionSpace::HessianRangeType HessianRangeType;

  enum { dimRange  = BaseType :: dimRange };
  enum { dimDomain = BaseType :: dimDomain };

  //! the right hand side data (default = 0)
  virtual void f(const DomainType& x,
                 RangeType& ret) const
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
  virtual void u(const DomainType& x,
                 RangeType& phi) const
  {
    assert( dimRange == 1 );
    phi[ 0 ] = 1;
    for( int i = 0; i < dimDomain; ++i )
      phi[ 0 ] *= std::cos( 2*M_PI*x[ i ] );
  }

  //! the jacobian of the exact solution
  virtual void uJacobian(const DomainType& x,
                         JacobianRangeType& ret) const
  {
    assert( dimRange == 1 );
    for( int i = 0; i < dimDomain; ++i )
    {
      ret[ 0 ][ i ] = -2*M_PI*std::sin( 2*M_PI*x[ i ] );
      for( int j = 1; j < dimDomain; ++j )
        ret[ 0 ][ i ] *= std::cos( 2*M_PI*x[ (i+j)%dimDomain ] );
    }
  }

  //! mass coefficient has to be 1 for this problem
  virtual void m(const DomainType& x, RangeType &m) const
  {
    m = RangeType(1);
  }

private:
  void hessian ( const DomainType &x, HessianRangeType &ret ) const
  {
    assert( dimRange == 1 );
    for( int i = 0; i < dimDomain; ++i )
    {
      for( int j = 0; j < dimDomain; ++j )
      {
        if ( j == i )
        {
          ret[ 0 ][ i ][ j ] = -4*M_PI*M_PI;
          for( int k = 0; k < dimDomain; ++k )
            ret[ 0 ][ i ][ j ] *= std::cos( 2*M_PI*x[ k ] );
        }
        else
        {
          ret[ 0 ][ i ][ j ] = 4*M_PI*M_PI;
          for( int k = 0; k < dimDomain; ++k )
            if ( k != i && k != j )
              ret[ 0 ][ i ][ j ] *= std::cos( 2*M_PI*x[ k ] );
            else
              ret[ 0 ][ i ][ j ] *= std::sin( 2*M_PI*x[ k ] );
        }
      }
    }
  }
};

// a reentrant corner problem with a 270 degree corner
template <class FunctionSpace>
class ReentrantCorner : public ProblemInterface < FunctionSpace >
{
  typedef ProblemInterface < FunctionSpace >  BaseType;

  const double lambda_;

public:
  typedef typename BaseType :: RangeType            RangeType;
  typedef typename BaseType :: DomainType           DomainType;
  typedef typename BaseType :: JacobianRangeType    JacobianRangeType;
  typedef typename BaseType :: DiffusionTensorType  DiffusionTensorType;

  enum { dimRange  = BaseType :: dimRange };
  enum { dimDomain = BaseType :: dimDomain };

  ReentrantCorner() : lambda_( 180./270.) {}

  //! the right hand side data (default = 0)
  virtual void f(const DomainType& x,
                 RangeType& phi) const
  {
    phi = 0;
  }

  //! the exact solution
  virtual void u(const DomainType& x,
                 RangeType& ret) const
  {
    double r2 = x.two_norm2();
    double phi = argphi(x[0],x[1]);
    ret = std::pow(r2, lambda_ * 0.5) * std::sin(lambda_*phi);
  }

  //! the jacobian of the exact solution
  virtual void uJacobian(const DomainType& x,
                         JacobianRangeType& ret) const
  {
    double grad[ 2 ] = { 0.0, 0.0 };

    double r2 = x.two_norm2();
    double phi=argphi(x[0],x[1]);
    double r2dx=2.*x[0];
    double r2dy=2.*x[1];
    double phidx=-x[1]/r2;
    double phidy=x[0]/r2;
    double lambdaPow = (lambda_*0.5)*pow(r2,lambda_*0.5-1.);
    grad[0]= lambdaPow * r2dx * sin(lambda_ * phi)
             + lambda_ * cos( lambda_ * phi) * phidx * pow(r2,lambda_ * 0.5);
    grad[1]= lambdaPow * r2dy * sin(lambda_ * phi)
             + lambda_ * cos( lambda_ * phi) * phidy * pow(r2,lambda_*0.5);

    assert( grad[0] == grad[0] );
    assert( grad[1] == grad[1] );

    assert( dimRange == 1 );
    for( int i = 0; i < dimDomain; ++i )
    {
      ret[ 0 ][ i ] = grad[ i ];
    }
  }

  //! return true if Dirichlet boundary is present (default is true)
  virtual bool hasDirichletBoundary () const
  {
    return true ;
  }

  //! return true if given point belongs to the Dirichlet boundary (default is true)
  virtual bool isDirichletPoint( const DomainType& x ) const
  {
    return true ;
  }
protected:
  /** \brief proper implementation of atan(x,y)
   */
  inline double argphi(double x,double y) const
  {
    double phi = std::arg(std::complex<double>(x,y));
    if (y<0) phi+=2.*M_PI;
    return phi;
  }

};

template <class FunctionSpace>
class CurvedRidges : public ProblemInterface < FunctionSpace >
{
  typedef ProblemInterface < FunctionSpace >  BaseType;
public:
  typedef typename BaseType :: RangeType            RangeType;
  typedef typename BaseType :: DomainType           DomainType;
  typedef typename BaseType :: JacobianRangeType    JacobianRangeType;
  typedef typename BaseType :: DiffusionTensorType  DiffusionTensorType;

  enum { dimRange  = BaseType :: dimRange };
  enum { dimDomain = BaseType :: dimDomain };

  //! the right hand side data (default = 0)
  virtual void f(const DomainType& p,
                 RangeType& phi) const
  {
    double q = p[ 0 ];
    for (int i=1; i<dimDomain; ++i)
    {
      q += std::sin(10*p[ i ]+5*p[ 0 ]*p[ 0 ]);
    }
    const double u = std::exp(q);
    double t1 = 1, t2 = 0, t3 = 0;
    for (int i=1; i<dimDomain; ++i)
    {
      t1 += std::cos(10*p[ i ]+5*p[ 0 ]*p[ 0 ]) * 10 * p[ 0 ];
      t2 += 10*std::cos(10*p[ i ]+5*p[ 0 ]*p[ 0 ]) -
      100*std::sin(10*p[ i ]+5*p[ 0 ]*p[ 0 ]) * p[ 0 ]*p[ 0 ];
      t3 += 100*std::cos(10*p[ i ]+5*p[ 0 ]*p[ 0 ])*std::cos(10*p[ i ]+5*p[ 0 ]*p[ 0 ]) -
      100*std::sin(10*p[ i ]+5*p[ 0 ]*p[ 0 ]);
    }
    t1 = t1*t1;

    phi = -u*(t1+t2+t3);
  }

  //! the exact solution
  virtual void u(const DomainType& p,
                 RangeType& phi) const
  {
    double q = p[ 0 ];
    for (int i=1; i<dimDomain; ++i)
    {
      q += std::sin(10*p[ i ]+5*p[ 0 ]*p[ 0 ]);
    }
    const double exponential = std::exp(q);
    phi = exponential;
  }

  //! the jacobian of the exact solution
  virtual void uJacobian(const DomainType& x,
                         JacobianRangeType& value) const
  {
    DUNE_THROW(Dune::NotImplemented,"still todo");
  }

  //! return true if Dirichlet boundary is present (default is true)
  virtual bool hasDirichletBoundary () const
  {
    return true ;
  }

  //! return true if given point belongs to the Dirichlet boundary (default is true)
  virtual bool isDirichletPoint( const DomainType& x ) const
  {
    return true ;
  }
};

#endif // #ifndef POISSON_HH
