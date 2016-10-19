/**************************************************************************

  The dune-fem module is a module of DUNE (see www.dune-project.org).
  It is based on the dune-grid interface library
  extending the grid interface by a number of discretization algorithms
  for solving non-linear systems of partial differential equations.

  Copyright (C) 2003 - 2014 Robert Kloefkorn
  Copyright (C) 2003 - 2010 Mario Ohlberger
  Copyright (C) 2004 - 2014 Andreas Dedner
  Copyright (C) 2005        Adrian Burri
  Copyright (C) 2005 - 2014 Mirko Kraenkel
  Copyright (C) 2006 - 2014 Christoph Gersbacher
  Copyright (C) 2006 - 2014 Martin Nolte
  Copyright (C) 2011 - 2014 Tobias Malkmus
  Copyright (C) 2012 - 2014 Stefan Girke
  Copyright (C) 2013 - 2014 Claus-Justus Heine
  Copyright (C) 2013 - 2014 Janick Gerstenberger
  Copyright (C) 2013        Sven Kaulman
  Copyright (C) 2013        Tom Ranner


  The dune-fem module is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of
  the License, or (at your option) any later version.

  The dune-fem module is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along
  with this program; if not, write to the Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

**************************************************************************/
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
  //! the Dirichlet boundary data (default calls u)
  virtual void g(const DomainType& x,
                 RangeType& value) const
  {
    value = RangeType( 0 );
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
    phi = 4*dimDomain*(M_PI*M_PI);
    for( int i = 0; i < dimDomain; ++i )
      phi *= std::sin( 2*M_PI*x[ i ] );
  }

  //! the exact solution
  virtual void u(const DomainType& x,
                 RangeType& phi) const
  {
    phi = 1;
    for( int i = 0; i < dimDomain; ++i )
      phi *= std::sin( 2*M_PI*x[ i ] );
    phi[0] += x[0]*x[0]-x[1]*x[1]+x[0]*x[1];
    // phi[0] += x[0]*x[1];
    // phi[0] += 0.5;
  }

  //! the jacobian of the exact solution
  virtual void uJacobian(const DomainType& x,
                         JacobianRangeType& ret) const
  {
    for( int r = 0; r < dimRange; ++ r )
    {
      for( int i = 0; i < dimDomain; ++i )
      {
        ret[ r ][ i ] = 2*M_PI*std::cos( 2*M_PI*x[ i ] );
        for( int j = 1; j < dimDomain; ++j )
          ret[ r ][ i ] *= std::sin( 2*M_PI*x[ (i+j)%dimDomain ] );
      }
    }
    ret[0][0] +=  2.*x[0]+x[1];
    ret[0][1] += -2.*x[1]+x[0];
    // ret[0][0] += x[1];
    // ret[0][1] += x[0];
  }
  //! mass coefficient has to be 1 for this problem
  virtual void m(const DomainType& x, RangeType &m) const
  {
    m = RangeType(0);
  }

  //! return true if given point belongs to the Dirichlet boundary (default is true)
  virtual bool isDirichletPoint( const DomainType& x ) const
  {
    return true;
  }
  virtual bool hasDirichletBoundary () const
  {
    return true ;
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
