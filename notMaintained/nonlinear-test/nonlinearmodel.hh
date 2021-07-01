#ifndef ELLIPTC_NONLINEARMODEL_HH
#define ELLIPTC_NONLINEARMODEL_HH

#include <cassert>
#include <cmath>

#include <dune/fem/solver/timeprovider.hh>
#include <dune/fem/io/parameter.hh>

#include "probleminterface.hh"

// DiffusionModel
// --------------

template< class FunctionSpace, class GridPart >
struct NonLinearModel
{
  typedef FunctionSpace FunctionSpaceType;
  typedef GridPart GridPartType;

  typedef typename FunctionSpaceType::DomainType DomainType;
  typedef typename FunctionSpaceType::RangeType RangeType;
  typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;

  typedef typename FunctionSpaceType::DomainFieldType DomainFieldType;
  typedef typename FunctionSpaceType::RangeFieldType RangeFieldType;

  static const int dimRange = FunctionSpaceType::dimRange;

  static const bool isLinear = false;
  static const bool isSymmetric = false;

  struct ProblemType : public ProblemInterface < FunctionSpace >
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
      JacobianRangeType grad;
      uJacobian(x,grad);
      phi *= (uVal[0]*uVal[0]+2.);
      phi -= (grad[0]*grad[0])*2.*uVal[0];
      phi += uVal[0]*uVal[0]*uVal[0] / 3.0;
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
  };

protected:
  enum FunctionId { rhs, bndN, bndD };
  template <FunctionId id>
  class FunctionWrapper;
public:
  typedef Dune::Fem::GridFunctionAdapter< FunctionWrapper<rhs>, GridPartType > RightHandSideType;
  typedef Dune::Fem::GridFunctionAdapter< FunctionWrapper<bndD>, GridPartType > DirichletBoundaryType;
  typedef Dune::Fem::GridFunctionAdapter< FunctionWrapper<bndN>, GridPartType > NeumanBoundaryType;

  //! constructor taking problem reference
  NonLinearModel( const ProblemType& problem, const GridPart &gridPart )
    : problem_( problem ),
      gridPart_(gridPart),
      rhs_(problem_),
      bndD_(problem_),
      bndN_(problem_),
      penalty_(Dune::Fem::Parameter::getValue<double>("dg.penalty", 0.0))
  {
  }

  template< class Entity, class Point >
  void source ( const Entity &entity,
                const Point &x,
                const RangeType &value,
                const JacobianRangeType &gradient,
                RangeType &flux ) const
  {
    flux[0] = value[0]*value[0]*value[0] / 3.0;
  }

  // the linearization of the source function
  template< class Entity, class Point >
  void linSource ( const RangeType& uBar,
                   const JacobianRangeType &gradientBar,
                   const Entity &entity,
                   const Point &x,
                   const RangeType &value,
                   const JacobianRangeType &gradient,
                   RangeType &flux ) const
  {
    flux[0] = uBar[0]*uBar[0]*value[0];
  }

  //! return the diffusive flux
  template< class Entity, class Point >
  void diffusiveFlux ( const Entity &entity,
                       const Point &x,
                       const RangeType &value,
                       const JacobianRangeType &gradient,
                       JacobianRangeType &flux ) const
  {
    flux = gradient;
    flux *= value[0]*value[0]+2.;
  }
  // linearization of diffusiveFlux
  template< class Entity, class Point >
  void linDiffusiveFlux ( const RangeType& uBar,
                          const JacobianRangeType& gradientBar,
                          const Entity &entity,
                          const Point &x,
                          const RangeType &value,
                          const JacobianRangeType &gradient,
                          JacobianRangeType &flux ) const
  {
    diffusiveFlux(entity,x,uBar,gradient,flux);
    flux.axpy(2.*uBar[0]*value[0],gradientBar);
  }

  template< class Entity, class Point >
  void alpha(const Entity &entity, const Point &x,
             const RangeType &value,
             RangeType &val) const
  {
    linAlpha(value,entity,x,value,val);
  }
  template< class Entity, class Point >
  void linAlpha(const RangeType &uBar,
                const Entity &entity, const Point &x,
                const RangeType &value,
                RangeType &val) const
  {
    val = RangeType(0);
  }

  //! exact some methods from the problem class
  bool hasDirichletBoundary () const
  {
    return false ;
  }
  bool hasNeumanBoundary () const
  {
    return false;
  }

  //! return true if given intersection belongs to the Dirichlet boundary -
  //! we test here if the center is a dirichlet point
  template <class Intersection>
  bool isDirichletIntersection( const Intersection& inter, Dune::FieldVector<bool,dimRange> &dirichletComponent ) const
  {
    return false;
  }

  // return Fem :: Function for Dirichlet boundary values
  DirichletBoundaryType dirichletBoundary( ) const
  {
    return DirichletBoundaryType( "boundary function", bndD_, gridPart_, 5 );
  }
  NeumanBoundaryType neumanBoundary( ) const
  {
    return NeumanBoundaryType( "boundary function", bndN_, gridPart_, 5 );
  }

  // return Fem :: Function for right hand side
  RightHandSideType rightHandSide(  ) const
  {
    return RightHandSideType( "right hand side", rhs_, gridPart_, 5 );
  }

  //penalty parameter
  double penalty() const
  {
    return penalty_;
  }
protected:
  template <FunctionId id>
  class FunctionWrapper : public Dune::Fem::Function< FunctionSpaceType, FunctionWrapper< id > >
  {
    const ProblemInterface<FunctionSpaceType>& impl_;
    public:
    FunctionWrapper( const ProblemInterface<FunctionSpaceType>& impl )
    : impl_( impl ) {}

    //! evaluate function
    void evaluate( const DomainType& x, RangeType& ret ) const
    {
      if( id == rhs )
      {
        // call right hand side of implementation
        impl_.f( x, ret );
      }
      else if( id == bndD )
      {
        // call dirichlet boudary data of implementation
        impl_.g( x, ret );
      }
      else if( id == bndN )
      {
        // call dirichlet boudary data of implementation
        impl_.n( x, ret );
      }
      else
      {
        DUNE_THROW(Dune::NotImplemented,"FunctionId not implemented");
      }
    }
  };

  const ProblemType& problem_;
  const GridPart &gridPart_;
  FunctionWrapper<rhs> rhs_;
  FunctionWrapper<bndD> bndD_;
  FunctionWrapper<bndN> bndN_;
  double penalty_;
};

#endif // #ifndef ELLIPTC_NONLINEARMODEL_HH
