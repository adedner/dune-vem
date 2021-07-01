#ifndef ELLIPTC_MODELINTERFACE_HH
#define ELLIPTC_MODELINTERFACE_HH

#include <cassert>
#include <cmath>

#include <dune/fem/solver/timeprovider.hh>
#include <dune/fem/io/parameter.hh>

template< class FunctionSpace, class GridPart >
struct DiffusionModelInterface
{
  typedef FunctionSpace FunctionSpaceType;
  typedef GridPart GridPartType;

  typedef typename FunctionSpaceType::DomainType DomainType;
  typedef typename FunctionSpaceType::RangeType RangeType;
  typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;

  typedef typename FunctionSpaceType::DomainFieldType DomainFieldType;
  typedef typename FunctionSpaceType::RangeFieldType RangeFieldType;

  static const int dimRange = FunctionSpaceType::dimRange;

  static const bool isLinear = true;
  static const bool isSymmetric = true;

  template< class Entity, class Point >
  void source ( const Entity &entity,
                const Point &x,
                const RangeType &value,
                const JacobianRangeType &gradient,
                RangeType &flux ) const
  {
    linSource( value, gradient, entity, x, value, gradient, flux );
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
    flux = 0;
  }
  //! return the diffusive flux
  template< class Entity, class Point >
  void diffusiveFlux ( const Entity &entity,
                       const Point &x,
                       const RangeType &value,
                       const JacobianRangeType &gradient,
                       JacobianRangeType &flux ) const
  {
    linDiffusiveFlux( value, gradient, entity, x, value, gradient, flux );
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
    // the flux is simply the identity
    flux = 0;
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
    val = 0;
  }
  //! extract some methods from the problem class for boundary traatment
  bool hasDirichletBoundary () const
  {
    return false;
  }
  bool hasNeumanBoundary () const
  {
    return false;
  }
  template <class Intersection>
  bool isDirichletIntersection( const Intersection& inter, Dune::FieldVector<bool,dimRange> &dirichletComponent ) const
  {
    return false;
  }
};

#endif // #ifndef ELLIPTC_MODEL_HH
