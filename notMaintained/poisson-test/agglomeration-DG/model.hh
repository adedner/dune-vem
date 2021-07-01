#ifndef ELLIPTC_MODEL_HH
#define ELLIPTC_MODEL_HH

#include <cassert>
#include <cmath>

#include <dune/fem/solver/timeprovider.hh>
#include <dune/fem/io/parameter.hh>
#include <dune/fem/quadrature/quadrature.hh>

#include "problems/interface.hh"


// DiffusionModel
// --------------

template< class FunctionSpace, class GridPart >
struct DiffusionModel
{
  typedef FunctionSpace FunctionSpaceType;
  typedef GridPart GridPartType;
  typedef typename GridPart::template Codim< 0 >::EntityType EntityType;

  static const int dimRange = FunctionSpaceType::dimRange;
  typedef typename FunctionSpaceType::DomainType DomainType;
  typedef typename FunctionSpaceType::RangeType RangeType;
  typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;

  typedef typename FunctionSpaceType::DomainFieldType DomainFieldType;
  typedef typename FunctionSpaceType::RangeFieldType RangeFieldType;

  typedef ProblemInterface< FunctionSpaceType > ProblemType ;

  static const bool isLinear = true;
  static const bool isSymmetric = true;

protected:
  enum FunctionId { rhs, bndD, bndN };
  template <FunctionId id>
  class FunctionWrapper;
public:
  typedef Dune::Fem::GridFunctionAdapter< FunctionWrapper<rhs>, GridPartType > RightHandSideType;
  typedef Dune::Fem::GridFunctionAdapter< FunctionWrapper<bndD>, GridPartType > DirichletBoundaryType;
  typedef Dune::Fem::GridFunctionAdapter< FunctionWrapper<bndN>, GridPartType > NeumanBoundaryType;

  //! constructor taking problem reference
  DiffusionModel( const ProblemType& problem, const GridPart &gridPart )
    : problem_( problem ),
      gridPart_(gridPart),
      rhs_(problem_),
      bndD_(problem_),
      bndN_(problem_),
      penalty_(Dune::Fem::Parameter::getValue<double>("dg.penalty"))
  {
  }

  bool init ( const EntityType &entity ) const
  {
    entity_ = &entity;
    return true;
  }
  const EntityType &entity () const
  {
    return *entity_;
  }
  template< class Point >
  void source ( const Point &x,
                const RangeType &value,
                const JacobianRangeType &gradient,
                RangeType &flux ) const
  {
    linSource( value, gradient, x, value, gradient, flux );
  }

  // the linearization of the source function
  template< class Point >
  void linSource ( const RangeType& uBar,
                   const JacobianRangeType &gradientBar,
                   const Point &x,
                   const RangeType &value,
                   const JacobianRangeType &gradient,
                   RangeType &flux ) const
  {
    const DomainType xGlobal = entity().geometry().global( Dune::Fem::coordinate( x ) );
    RangeType m;
    problem_.m(xGlobal,m);
    for (unsigned int i=0;i<flux.size();++i)
      flux[i] = m[i]*value[i];
  }
  //! return the diffusive flux
  template< class Point >
  void diffusiveFlux ( const Point &x,
                       const RangeType &value,
                       const JacobianRangeType &gradient,
                       JacobianRangeType &flux ) const
  {
    linDiffusiveFlux( value, gradient, x, value, gradient, flux );
  }
  // linearization of diffusiveFlux
  template< class Point >
  void linDiffusiveFlux ( const RangeType& uBar,
                          const JacobianRangeType& gradientBar,
                          const Point &x,
                          const RangeType &value,
                          const JacobianRangeType &gradient,
                          JacobianRangeType &flux ) const
  {
    const DomainType xGlobal = entity().geometry().global( coordinate( x ) );
    for( std::size_t i = 0; i < flux.rows; ++i )
      problem_.diffusionTensor( xGlobal ).mv( gradient[ i ], flux[ i ] );
  }

  template< class Point >
  void alpha(const Point &x,
             const RangeType &value,
             RangeType &val) const
  {
    linAlpha(value,x,value,val);
  }
  template< class Point >
  void linAlpha(const RangeType &uBar,
                const Point &x,
                const RangeType &value,
                RangeType &val) const
  {
    const DomainType xGlobal = entity().geometry().global( coordinate( x ) );
    RangeType alpha;
    problem_.alpha(xGlobal,alpha);
    for (unsigned int i=0;i<val.size();++i)
      val[i] = alpha[i]*value[i];
  }
  //! extract some methods from the problem class
  bool hasDirichletBoundary () const
  {
    return problem_.hasDirichletBoundary() ;
  }
  bool hasNeumanBoundary () const
  {
    return problem_.hasNeumanBoundary() ;
  }

  //! return true if given intersection belongs to the Dirichlet boundary -
  //! we test here if the center is a dirichlet point
  template <class Intersection>
  bool isDirichletIntersection( const Intersection& inter, Dune::FieldVector<int,dimRange> &dirichletComponent ) const
  {
    return problem_.isDirichletPoint( inter.geometry().center() ) ;
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

  template< class Entity, class Point >
  void g( const RangeType& uBar,
          const Entity &entity,
          const Point &x,
          RangeType &u ) const
  {
    const DomainType xGlobal = entity.geometry().global( Dune::Fem::coordinate( x ) );
    problem_.g( xGlobal, u );
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
  mutable const EntityType *entity_ = nullptr;
};

#endif // #ifndef ELLIPTC_MODEL_HH
