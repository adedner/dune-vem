#ifndef VEMELLIPTC_MODEL_HH
#define VEMELLIPTC_MODEL_HH

#include <cassert>
#include <cmath>

#include <dune/fem/solver/timeprovider.hh>
#include <dune/fem/io/parameter.hh>
#include <dune/fem/space/common/functionspace.hh>
#include <dune/fem/schemes/integrands.hh>
#include <dune/fem/function/common/gridfunctionadapter.hh>
#include <dune/fem/quadrature/cachingquadrature.hh>
#include <dune/fempy/quadrature/fempyquadratures.hh>

#define VirtualVEMDiffusionModelMethods(POINT) \
  virtual void source ( const POINT &x,\
                const DRangeType &value,\
                const DJacobianRangeType &gradient,\
                RRangeType &flux ) const = 0;\
  virtual void linSource ( const DRangeType& uBar,\
                   const DJacobianRangeType &gradientBar,\
                   const POINT &x,\
                   const DRangeType &value,\
                   const DJacobianRangeType &gradient,\
                   RRangeType &flux ) const = 0;\
  virtual void diffusiveFlux ( const POINT &x,\
                       const DRangeType &value,\
                       const DJacobianRangeType &gradient,\
                       RJacobianRangeType &flux ) const = 0;\
  virtual void linDiffusiveFlux ( const DRangeType& uBar,\
                          const DJacobianRangeType& gradientBar,\
                          const POINT &x,\
                          const DRangeType &value,\
                          const DJacobianRangeType &gradient,\
                          RJacobianRangeType &flux ) const = 0;\
  virtual void fluxDivergence( const POINT &x,\
                         const DRangeType &value,\
                         const DJacobianRangeType &jacobian,\
                         const DHessianRangeType &hessian,\
                         RRangeType &flux) const = 0;\
  virtual void alpha(const POINT &x,\
             const DRangeType &value,\
             RRangeType &val) const = 0;\
  virtual void linAlpha(const DRangeType &uBar,\
                const POINT &x,\
                const DRangeType &value,\
                RRangeType &val) const = 0;\
  virtual void dirichlet( int bndId, const POINT &x,\
                RRangeType &value) const = 0; \
  virtual RRangeType massStabilization( const POINT &x,\
                const DRangeType &value) const = 0; \
  virtual RRangeType linMassStabilization( const POINT &x,\
                const DRangeType &value) const = 0; \
  virtual RRangeType gradStabilization( const POINT &x,\
                const DRangeType &value) const = 0; \
  virtual RRangeType linGradStabilization( const POINT &x,\
                const DRangeType &value) const = 0;

#define WrapperVEMDiffusionModelMethods(POINT) \
  virtual void source ( const POINT &x,\
                const DRangeType &value,\
                const DJacobianRangeType &gradient,\
                RRangeType &flux ) const \
  { impl().source(x, value, gradient, flux); } \
  virtual void linSource ( const DRangeType& uBar,\
                   const DJacobianRangeType &gradientBar,\
                   const POINT &x,\
                   const DRangeType &value,\
                   const DJacobianRangeType &gradient,\
                   RRangeType &flux ) const \
  { impl().linSource(uBar, gradientBar, x, value, gradient, flux); } \
  virtual void diffusiveFlux ( const POINT &x,\
                       const DRangeType &value,\
                       const DJacobianRangeType &gradient,\
                       RJacobianRangeType &flux ) const \
  { impl().diffusiveFlux(x, value, gradient, flux); } \
  virtual void linDiffusiveFlux ( const DRangeType& uBar,\
                          const DJacobianRangeType& gradientBar,\
                          const POINT &x,\
                          const DRangeType &value,\
                          const DJacobianRangeType &gradient,\
                          RJacobianRangeType &flux ) const \
  { impl().linDiffusiveFlux(uBar, gradientBar, x, value, gradient, flux); } \
  virtual void fluxDivergence( const POINT &x,\
                         const DRangeType &value,\
                         const DJacobianRangeType &jacobian,\
                         const DHessianRangeType &hessian,\
                         RRangeType &flux) const \
  { impl().fluxDivergence(x, value, jacobian, hessian, flux); } \
  virtual void alpha(const POINT &x,\
             const DRangeType &value,\
             RRangeType &val) const \
  { impl().alpha(x, value, val); } \
  virtual void linAlpha(const DRangeType &uBar,\
                const POINT &x,\
                const DRangeType &value,\
                RRangeType &val) const \
  { impl().linAlpha(uBar, x, value, val); } \
  virtual void dirichlet( int bndId, const POINT &x,\
                RRangeType &value) const \
  { impl().dirichlet(bndId, x, value); } \
  virtual RRangeType massStabilization( const POINT &x,\
                const DRangeType &value) const \
  { return impl().massStabilization(x,value); } \
  virtual RRangeType linMassStabilization( const POINT &x,\
                const DRangeType &value) const \
  { return impl().linMassStabilization(x,value); } \
  virtual RRangeType gradStabilization( const POINT &x,\
                const DRangeType &value) const \
  { return impl().gradStabilization(x,value); } \
  virtual RRangeType linGradStabilization( const POINT &x,\
                const DRangeType &value) const \
  { return impl().linGradStabilization(x,value); }

template< class GridPart, int dimDomain, int dimRange=dimDomain, class RangeField = double >
struct VEMDiffusionModel
{
  typedef GridPart GridPartType;
  static const int dimD = dimDomain;
  static const int dimR = dimRange;
  typedef VEMDiffusionModel<GridPartType, dimD, dimR, RangeField> ModelType;
  typedef RangeField RangeFieldType;

  typedef Dune::Fem::FunctionSpace< double, RangeFieldType,
              GridPart::dimensionworld, dimD > DFunctionSpaceType;
  typedef Dune::Fem::FunctionSpace< double, RangeFieldType,
              GridPart::dimensionworld, dimR > RFunctionSpaceType;
  typedef typename DFunctionSpaceType::DomainType DomainType;
  typedef typename DFunctionSpaceType::RangeType DRangeType;
  typedef typename DFunctionSpaceType::JacobianRangeType DJacobianRangeType;
  typedef typename DFunctionSpaceType::HessianRangeType DHessianRangeType;
  typedef typename DFunctionSpaceType::DomainFieldType DDomainFieldType;
  typedef typename RFunctionSpaceType::RangeType RRangeType;
  typedef typename RFunctionSpaceType::JacobianRangeType RJacobianRangeType;
  typedef typename RFunctionSpaceType::HessianRangeType RHessianRangeType;
  typedef typename RFunctionSpaceType::DomainFieldType rDomainFieldType;

  typedef typename GridPartType::template Codim<0>::EntityType EntityType;
  typedef typename GridPartType::IntersectionType IntersectionType;
  typedef typename EntityType::Geometry::LocalCoordinate LocalDomainType;

  // quadrature points from dune-fempy quadratures
  template <class F,int d>
  using Traits = Dune::FemPy::FempyQuadratureTraits<F,d>;
  typedef typename Dune::Fem::CachingQuadrature< GridPartType, 0, Traits >::
                   QuadraturePointWrapperType Point;
  typedef typename Dune::Fem::CachingQuadrature< GridPartType, 1, Traits >::
                   QuadraturePointWrapperType IntersectionPoint;
  typedef typename Dune::Fem::ElementQuadrature< GridPartType, 0, Traits >::
                   QuadraturePointWrapperType ElementPoint;
  typedef typename Dune::Fem::ElementQuadrature< GridPartType, 1, Traits >::
                   QuadraturePointWrapperType ElementIntersectionPoint;

  // quadrature points from dune-fem quadratures
  typedef typename Dune::Fem::CachingQuadrature< GridPartType, 0 >::
                   QuadraturePointWrapperType OriginalPoint;
  typedef typename Dune::Fem::CachingQuadrature< GridPartType, 1 >::
                   QuadraturePointWrapperType OriginalIntersectionPoint;
  typedef typename Dune::Fem::ElementQuadrature< GridPartType, 0 >::
                   QuadraturePointWrapperType OriginalElementPoint;
  typedef typename Dune::Fem::ElementQuadrature< GridPartType, 1 >::
                   QuadraturePointWrapperType OriginalElementIntersectionPoint;

  /*
  static const bool isLinear;
  static const bool isSymmetric;
  */

public:
  VEMDiffusionModel( )
  { }
  virtual ~VEMDiffusionModel() {}

  virtual std::string name() const = 0;

  virtual bool init( const EntityType &entity) const = 0;
  virtual void unbind() const = 0;

  // virtual methods for fempy quadratures
  VirtualVEMDiffusionModelMethods(Point)
  VirtualVEMDiffusionModelMethods(ElementPoint)
  VirtualVEMDiffusionModelMethods(IntersectionPoint)
  VirtualVEMDiffusionModelMethods(ElementIntersectionPoint)

  // virtual methods for fem quadratures
  VirtualVEMDiffusionModelMethods(OriginalPoint)
  VirtualVEMDiffusionModelMethods(OriginalElementPoint)
  VirtualVEMDiffusionModelMethods(OriginalIntersectionPoint)
  VirtualVEMDiffusionModelMethods(OriginalElementIntersectionPoint)

  VirtualVEMDiffusionModelMethods(LocalDomainType)

  typedef std::array<int, dimR> DirichletComponentType;
  virtual bool hasDirichletBoundary () const = 0;
  virtual bool hasNeumanBoundary () const = 0;
  virtual bool isDirichletIntersection( const IntersectionType& inter, DirichletComponentType &dirichletComponent ) const = 0;
};

template < class ModelImpl >
struct DiffusionModelWrapper : public VEMDiffusionModel<typename ModelImpl::GridPartType, ModelImpl::dimD, ModelImpl::dimR,
                                      typename ModelImpl::RRangeFieldType>
{
  typedef ModelImpl Impl;
  typedef typename ModelImpl::GridPartType GridPartType;
  static const int dimD  = ModelImpl::dimD;
  static const int dimR  = ModelImpl::dimR;
  typedef VEMDiffusionModel<GridPartType, dimD, dimR, typename ModelImpl::RRangeFieldType> Base;

  typedef typename Base::Point Point;
  typedef typename Base::IntersectionPoint IntersectionPoint;
  typedef typename Base::ElementPoint ElementPoint;
  typedef typename Base::ElementIntersectionPoint ElementIntersectionPoint;
  typedef typename Base::OriginalPoint OriginalPoint;
  typedef typename Base::OriginalIntersectionPoint OriginalIntersectionPoint;
  typedef typename Base::OriginalElementPoint      OriginalElementPoint;
  typedef typename Base::OriginalElementIntersectionPoint OriginalElementIntersectionPoint;
  typedef typename Base::LocalDomainType LocalDomainType;
  typedef typename Base::DomainType DomainType;
  typedef typename Base::DRangeType DRangeType;
  typedef typename Base::DJacobianRangeType DJacobianRangeType;
  typedef typename Base::DHessianRangeType DHessianRangeType;
  typedef typename Base::RRangeType RRangeType;
  typedef typename Base::RJacobianRangeType RJacobianRangeType;
  typedef typename Base::RHessianRangeType RHessianRangeType;
  typedef typename Base::EntityType EntityType;
  typedef typename Base::IntersectionType IntersectionType;

  template< class... Args, std::enable_if_t< std::is_constructible< ModelImpl, Args &&... >::value, int > = 0 >
  explicit DiffusionModelWrapper ( Args &&... args )
    : impl_( std::forward< Args >( args )... )
  {}

  ~DiffusionModelWrapper()
  {
  }

  // virtual methods for fempy quadratures
  WrapperVEMDiffusionModelMethods(Point);
  WrapperVEMDiffusionModelMethods(ElementPoint);
  WrapperVEMDiffusionModelMethods(IntersectionPoint);
  WrapperVEMDiffusionModelMethods(ElementIntersectionPoint);

  // virtual methods for fem quadratures
  WrapperVEMDiffusionModelMethods(OriginalPoint);
  WrapperVEMDiffusionModelMethods(OriginalElementPoint);
  WrapperVEMDiffusionModelMethods(OriginalIntersectionPoint);
  WrapperVEMDiffusionModelMethods(OriginalElementIntersectionPoint);

  WrapperVEMDiffusionModelMethods(LocalDomainType);

  // other virtual functions
  virtual std::string name() const
  {
    return impl().name();
  }
  typedef std::array<int, dimR> DirichletComponentType;
  virtual bool hasDirichletBoundary () const
  {
    return impl().hasDirichletBoundary();
  }
  virtual bool hasNeumanBoundary () const
  {
    return impl().hasNeumanBoundary();
  }
  virtual bool isDirichletIntersection( const IntersectionType& inter, DirichletComponentType &dirichletComponent ) const
  {
    return impl().isDirichletIntersection(inter, dirichletComponent);
  }
  virtual bool init( const EntityType &entity) const
  {
    return impl().init(entity);
  }
  virtual void unbind() const
  {
    return impl().unbind();
  }
  const ModelImpl &impl() const
  {
    return impl_;
  }
  ModelImpl &impl()
  {
    return impl_;
  }
  private:
  ModelImpl impl_;
};


namespace Dune
{

  namespace Fem
  {

    template< class Integrands >
    struct FullVemIntegrands : public FullIntegrands<Integrands>
    {
      typedef FullIntegrands<Integrands> Base;
      template< class... Args >
      explicit FullVemIntegrands ( Args &&... args )
        : Base( std::forward< Args >( args )... )
      {}
      using Base::integrands;
      template <class Point, class DRangeType>
      auto hessStabilization ( const Point &x, const DRangeType &u ) const
      {
        return integrands().hessStabilization(x,u);
      }
      template <class Point, class DRangeType>
      auto linHessStabilization ( const Point &x, const DRangeType &u ) const
      {
        return integrands().linHessStabilization(x,u);
      }
      template <class Point, class DRangeType>
      auto gradStabilization ( const Point &x, const DRangeType &u ) const
      {
        return integrands().gradStabilization(x,u);
      }
      template <class Point, class DRangeType>
      auto linGradStabilization ( const Point &x, const DRangeType &u ) const
      {
        return integrands().linGradStabilization(x,u);
      }
      template <class Point, class DRangeType>
      auto massStabilization ( const Point &x, const DRangeType &u ) const
      {
        return integrands().massStabilization(x,u);
      }
      template <class Point, class DRangeType>
      auto linMassStabilization ( const Point &x, const DRangeType &u ) const
      {
        return integrands().linMassStabilization(x,u);
      }
    };

    template< class GridPart, class DomainValue, class RangeValue = DomainValue >
    class VirtualizedVemIntegrands
    {
      typedef VirtualizedVemIntegrands< GridPart, DomainValue, RangeValue > This;

    public:
      typedef GridPart GridPartType;
      typedef DomainValue DomainValueType;
      typedef RangeValue RangeValueType;

      typedef typename GridPartType::template Codim< 0 >::EntityType EntityType;
      typedef typename GridPartType::IntersectionType IntersectionType;

      using RRangeType = typename detail::GetDimRange<std::tuple_element_t<0,RangeValueType>>::type;
      using DRangeType = typename detail::GetDimRange<std::tuple_element_t<0,DomainValueType>>::type;
      typedef std::array<int,RRangeType::dimension> DirichletComponentType;
      typedef typename EntityType::Geometry::LocalCoordinate DomainType;

    private:
      typedef typename EntityType::Geometry::LocalCoordinate LocalCoordinateType;

      typedef FemPy::CachingPoint< GridPart, LocalCoordinateType, 0 > InteriorCachingPointType;
      typedef FemPy::ElementPoint< GridPart, LocalCoordinateType, 0 > InteriorElementPointType;
      typedef FemPy::CachingPoint< GridPart, LocalCoordinateType, 1 > SurfaceCachingPointType;
      typedef FemPy::ElementPoint< GridPart, LocalCoordinateType, 1 > SurfaceElementPointType;

      template< class QP >
      static Fem::QuadraturePointWrapper< QP > asQP ( const QP &qp )
      {
        return static_cast< Fem::QuadraturePointWrapper< QP > >( qp );
      }

      template< class R >
      using Linearization = std::function< R( const DomainValueType & ) >;

      template< class R >
      using LinearizationPair = std::pair< Linearization< std::pair< R, R > >, Linearization< std::pair< R, R > > >;

      struct Interface
      {
        virtual ~Interface ()  {}
        virtual Interface *clone () const = 0;

        virtual bool init ( const EntityType &entity ) = 0;
        virtual bool init ( const IntersectionType &intersection ) = 0;
        virtual void unbind() = 0;

        virtual bool hasInterior () const = 0;
        virtual RangeValueType interior ( const InteriorCachingPointType &x, const DomainValueType &u ) const = 0;
        virtual RangeValueType interior ( const InteriorElementPointType &x, const DomainValueType &u ) const = 0;
        virtual Linearization< RangeValueType > linearizedInterior ( const InteriorCachingPointType &x, const DomainValueType &u ) const = 0;
        virtual Linearization< RangeValueType > linearizedInterior ( const InteriorElementPointType &x, const DomainValueType &u ) const = 0;

        virtual bool hasBoundary () const = 0;
        virtual RangeValueType boundary ( const SurfaceCachingPointType &x, const DomainValueType &u ) const = 0;
        virtual RangeValueType boundary ( const SurfaceElementPointType &x, const DomainValueType &u ) const = 0;
        virtual Linearization< RangeValueType > linearizedBoundary ( const SurfaceCachingPointType &x, const DomainValueType &u ) const = 0;
        virtual Linearization< RangeValueType > linearizedBoundary ( const SurfaceElementPointType &x, const DomainValueType &u ) const = 0;

        virtual bool hasSkeleton () const = 0;
        virtual std::pair< RangeValueType, RangeValueType > skeleton ( const SurfaceCachingPointType &xIn, const DomainValueType &uIn, const SurfaceCachingPointType &xOut, const DomainValueType &uOut ) const = 0;
        virtual std::pair< RangeValueType, RangeValueType > skeleton ( const SurfaceElementPointType &xIn, const DomainValueType &uIn, const SurfaceElementPointType &xOut, const DomainValueType &uOut ) const = 0;
        virtual LinearizationPair< RangeValueType > linearizedSkeleton ( const SurfaceCachingPointType &xIn, const DomainValueType &uIn, const SurfaceCachingPointType &xOut, const DomainValueType &uOut ) const = 0;
        virtual LinearizationPair< RangeValueType > linearizedSkeleton ( const SurfaceElementPointType &xIn, const DomainValueType &uIn, const SurfaceElementPointType &xOut, const DomainValueType &uOut ) const = 0;

        virtual bool hasDirichletBoundary () const = 0;
        virtual bool isDirichletIntersection( const IntersectionType& inter, DirichletComponentType &dirichletComponent ) = 0;
        virtual void dirichlet( int bndId, const DomainType &x,RRangeType &value) const = 0;

        virtual RRangeType hessStabilization ( const DomainType &x, const DRangeType &u ) const = 0;
        virtual RRangeType linHessStabilization ( const DomainType &x, const DRangeType &u ) const = 0;
        virtual RRangeType gradStabilization ( const DomainType &x, const DRangeType &u ) const = 0;
        virtual RRangeType linGradStabilization ( const DomainType &x, const DRangeType &u ) const = 0;
        virtual RRangeType massStabilization ( const DomainType &x, const DRangeType &u ) const = 0;
        virtual RRangeType linMassStabilization ( const DomainType &x, const DRangeType &u ) const = 0;
      };

      template< class Impl >
      struct Implementation final
        : public Interface
      {
        Implementation ( Impl impl ) : impl_( std::move( impl ) ) {}
        virtual Interface *clone () const override { return new Implementation( *this ); }

        virtual bool init ( const EntityType &entity ) override { return impl().init( entity ); }
        virtual bool init ( const IntersectionType &intersection ) override { return impl().init( intersection ); }
        virtual void unbind() override {impl().unbind(); }

        virtual bool hasInterior () const override { return impl().hasInterior(); }
        virtual RangeValueType interior ( const InteriorCachingPointType &x, const DomainValueType &u ) const override { return impl().interior( asQP( x ), u ); }
        virtual RangeValueType interior ( const InteriorElementPointType &x, const DomainValueType &u ) const override { return impl().interior( asQP( x ), u ); }
        virtual Linearization< RangeValueType > linearizedInterior ( const InteriorCachingPointType &x, const DomainValueType &u ) const override { return impl().linearizedInterior( asQP( x ), u ); }
        virtual Linearization< RangeValueType > linearizedInterior ( const InteriorElementPointType &x, const DomainValueType &u ) const override { return impl().linearizedInterior( asQP( x ), u ); }

        virtual bool hasBoundary () const override { return impl().hasBoundary(); }
        virtual RangeValueType boundary ( const SurfaceCachingPointType &x, const DomainValueType &u ) const override { return impl().boundary( asQP( x ), u ); }
        virtual RangeValueType boundary ( const SurfaceElementPointType &x, const DomainValueType &u ) const override { return impl().boundary( asQP( x ), u ); }
        virtual Linearization< RangeValueType > linearizedBoundary ( const SurfaceCachingPointType &x, const DomainValueType &u ) const override { return impl().linearizedBoundary( asQP( x ), u ); }
        virtual Linearization< RangeValueType > linearizedBoundary ( const SurfaceElementPointType &x, const DomainValueType &u ) const override { return impl().linearizedBoundary( asQP( x ), u ); }

        virtual bool hasSkeleton () const override { return impl().hasSkeleton(); }
        virtual std::pair< RangeValueType, RangeValueType > skeleton ( const SurfaceCachingPointType &xIn, const DomainValueType &uIn, const SurfaceCachingPointType &xOut, const DomainValueType &uOut ) const override { return impl().skeleton( asQP( xIn ), uIn, asQP( xOut ), uOut ); }
        virtual std::pair< RangeValueType, RangeValueType > skeleton ( const SurfaceElementPointType &xIn, const DomainValueType &uIn, const SurfaceElementPointType &xOut, const DomainValueType &uOut ) const override { return impl().skeleton( asQP( xIn ), uIn, asQP( xOut ), uOut ); }
        virtual LinearizationPair< RangeValueType > linearizedSkeleton ( const SurfaceCachingPointType &xIn, const DomainValueType &uIn, const SurfaceCachingPointType &xOut, const DomainValueType &uOut ) const override { return impl().linearizedSkeleton( asQP( xIn ), uIn, asQP( xOut ), uOut ); }
        virtual LinearizationPair< RangeValueType > linearizedSkeleton ( const SurfaceElementPointType &xIn, const DomainValueType &uIn, const SurfaceElementPointType &xOut, const DomainValueType &uOut ) const override { return impl().linearizedSkeleton( asQP( xIn ), uIn, asQP( xOut ), uOut ); }

        virtual bool hasDirichletBoundary () const override { return impl().hasDirichletBoundary(); }
        virtual bool isDirichletIntersection( const IntersectionType& inter, DirichletComponentType &dirichletComponent ) override { return impl().isDirichletIntersection(inter,dirichletComponent); }
        virtual void dirichlet( int bndId, const DomainType &x,RRangeType &value) const override { impl().dirichlet(bndId,x,value); }

        virtual RRangeType hessStabilization ( const DomainType &x, const DRangeType &u ) const override { return impl().hessStabilization(x,u); }
        virtual RRangeType linHessStabilization ( const DomainType &x, const DRangeType &u ) const override { return impl().linHessStabilization(x,u); }
        virtual RRangeType gradStabilization ( const DomainType &x, const DRangeType &u ) const override { return impl().gradStabilization(x,u); }
        virtual RRangeType linGradStabilization ( const DomainType &x, const DRangeType &u ) const override { return impl().linGradStabilization(x,u); }
        virtual RRangeType massStabilization ( const DomainType &x, const DRangeType &u ) const override { return impl().massStabilization(x,u); }
        virtual RRangeType linMassStabilization ( const DomainType &x, const DRangeType &u ) const override { return impl().linMassStabilization(x,u); }

      private:
        const auto &impl () const { return std::cref( impl_ ).get(); }
        auto &impl () { return std::ref( impl_ ).get(); }

        Impl impl_;
      };

      template< class Integrands >
      using isVirtualized = std::is_same< std::decay_t< decltype( std::ref( std::declval< Integrands & >() ).get() ) >, This >;

    public:
      template< class Integrands, std::enable_if_t< IntegrandsTraits< std::decay_t< Integrands > >::isFull && !isVirtualized< Integrands >::value, int > = 0 >
      explicit VirtualizedVemIntegrands ( Integrands integrands )
        : impl_( new Implementation< Integrands >( std::move( integrands ) ) )
      {}

      template< class Integrands, std::enable_if_t< !IntegrandsTraits< Integrands >::isFull && !isVirtualized< Integrands >::value, int > = 0 >
      explicit VirtualizedVemIntegrands ( Integrands integrands )
        : VirtualizedVemIntegrands( FullVemIntegrands< std::decay_t< Integrands > >( std::move( integrands ) ) )
      {}

      VirtualizedVemIntegrands ( const This &other ) : impl_( other ? other.impl().clone() : nullptr ) {}
      VirtualizedVemIntegrands ( This && ) = default;

      VirtualizedVemIntegrands &operator= ( const This &other ) { impl_.reset( other ? other.impl().clone() : nullptr ); }
      VirtualizedVemIntegrands &operator= ( This && ) = default;

      explicit operator bool () const { return static_cast< bool >( impl_ ); }

      bool init ( const EntityType &entity ) { return impl().init( entity ); }
      bool init ( const IntersectionType &intersection ) { return impl().init( intersection ); }
      void unbind() { impl().unbind(); }

      bool hasInterior () const { return impl().hasInterior(); }

      template< class Quadrature, std::enable_if_t< std::is_convertible< Quadrature, Fem::CachingInterface >::value, int > = 0 >
      RangeValueType interior ( const Fem::QuadraturePointWrapper< Quadrature > &x, const DomainValueType &u ) const
      {
        return impl().interior( InteriorCachingPointType( x ), u );
      }

      template< class Quadrature, std::enable_if_t< !std::is_convertible< Quadrature, Fem::CachingInterface >::value, int > = 0 >
      RangeValueType interior ( const Fem::QuadraturePointWrapper< Quadrature > &x, const DomainValueType &u ) const
      {
        return impl().interior( InteriorElementPointType( x ), u );
      }

      template< class Quadrature, std::enable_if_t< std::is_convertible< Quadrature, Fem::CachingInterface >::value, int > = 0 >
      auto linearizedInterior ( const Fem::QuadraturePointWrapper< Quadrature > &x, const DomainValueType &u ) const
      {
        return impl().linearizedInterior( InteriorCachingPointType( x ), u );
      }

      template< class Quadrature, std::enable_if_t< !std::is_convertible< Quadrature, Fem::CachingInterface >::value, int > = 0 >
      auto linearizedInterior ( const Fem::QuadraturePointWrapper< Quadrature > &x, const DomainValueType &u ) const
      {
        return impl().linearizedInterior( InteriorElementPointType( x ), u );
      }

      bool hasBoundary () const { return impl().hasBoundary(); }

      template< class Quadrature, std::enable_if_t< std::is_convertible< Quadrature, Fem::CachingInterface >::value, int > = 0 >
      RangeValueType boundary ( const Fem::QuadraturePointWrapper< Quadrature > &x, const DomainValueType &u ) const
      {
        return impl().boundary( SurfaceCachingPointType( x ), u );
      }

      template< class Quadrature, std::enable_if_t< !std::is_convertible< Quadrature, Fem::CachingInterface >::value, int > = 0 >
      RangeValueType boundary ( const Fem::QuadraturePointWrapper< Quadrature > &x, const DomainValueType &u ) const
      {
        return impl().boundary( SurfaceElementPointType( x ), u );
      }

      template< class Quadrature, std::enable_if_t< std::is_convertible< Quadrature, Fem::CachingInterface >::value, int > = 0 >
      auto linearizedBoundary ( const Fem::QuadraturePointWrapper< Quadrature > &x, const DomainValueType &u ) const
      {
        return impl().linearizedBoundary( SurfaceCachingPointType( x ), u );
      }

      template< class Quadrature, std::enable_if_t< !std::is_convertible< Quadrature, Fem::CachingInterface >::value, int > = 0 >
      auto linearizedBoundary ( const Fem::QuadraturePointWrapper< Quadrature > &x, const DomainValueType &u ) const
      {
        return impl().linearizedBoundary( SurfaceElementPointType( x ), u );
      }

      bool hasSkeleton () const { return impl().hasSkeleton(); }

      template< class Quadrature, std::enable_if_t< std::is_convertible< Quadrature, Fem::CachingInterface >::value, int > = 0 >
      std::pair< RangeValueType, RangeValueType > skeleton ( const Fem::QuadraturePointWrapper< Quadrature > &xIn, const DomainValueType &uIn, const Fem::QuadraturePointWrapper< Quadrature > &xOut, const DomainValueType &uOut ) const
      {
        return impl().skeleton( SurfaceCachingPointType( xIn ), uIn, SurfaceCachingPointType( xOut ), uOut );
      }

      template< class Quadrature, std::enable_if_t< !std::is_convertible< Quadrature, Fem::CachingInterface >::value, int > = 0 >
      std::pair< RangeValueType, RangeValueType > skeleton ( const Fem::QuadraturePointWrapper< Quadrature > &xIn, const DomainValueType &uIn, const Fem::QuadraturePointWrapper< Quadrature > &xOut, const DomainValueType &uOut ) const
      {
        return impl().skeleton( SurfaceElementPointType( xIn ), uIn, SurfaceElementPointType( xOut ), uOut );
      }

      template< class Quadrature, std::enable_if_t< std::is_convertible< Quadrature, Fem::CachingInterface >::value, int > = 0 >
      auto linearizedSkeleton ( const Fem::QuadraturePointWrapper< Quadrature > &xIn, const DomainValueType &uIn, const Fem::QuadraturePointWrapper< Quadrature > &xOut, const DomainValueType &uOut ) const
      {
        return impl().linearizedSkeleton( SurfaceCachingPointType( xIn ), uIn, SurfaceCachingPointType( xOut ), uOut );
      }

      template< class Quadrature, std::enable_if_t< !std::is_convertible< Quadrature, Fem::CachingInterface >::value, int > = 0 >
      auto linearizedSkeleton ( const Fem::QuadraturePointWrapper< Quadrature > &xIn, const DomainValueType &uIn, const Fem::QuadraturePointWrapper< Quadrature > &xOut, const DomainValueType &uOut ) const
      {
        return impl().linearizedSkeleton( SurfaceElementPointType( xIn ), uIn, SurfaceElementPointType( xOut ), uOut );
      }

      bool hasDirichletBoundary () const
      {
        return impl().hasDirichletBoundary();
      }
      bool isDirichletIntersection( const IntersectionType& inter, DirichletComponentType &dirichletComponent )
      {
        return impl().isDirichletIntersection(inter,dirichletComponent);
      }
      void dirichlet( int bndId, const DomainType &x,RRangeType &value) const
      {
        return impl().dirichlet(bndId,x,value);
      }

      RRangeType hessStabilization ( const DomainType &x, const DRangeType &u ) const
      {
        return impl().hessStabilization(x,u);
      }
      RRangeType linHessStabilization ( const DomainType &x, const DRangeType &u ) const
      {
        return impl().linHessStabilization(x,u);
      }
      RRangeType gradStabilization ( const DomainType &x, const DRangeType &u ) const
      {
        return impl().gradStabilization(x,u);
      }
      RRangeType linGradStabilization ( const DomainType &x, const DRangeType &u ) const
      {
        return impl().linGradStabilization(x,u);
      }
      RRangeType massStabilization ( const DomainType &x, const DRangeType &u ) const
      {
        return impl().massStabilization(x,u);
      }
      RRangeType linMassStabilization ( const DomainType &x, const DRangeType &u ) const
      {
        return impl().linMassStabilization(x,u);
      }

      auto get() const { return impl_.get(); }
    private:
      const Interface &impl () const { assert( impl_ ); return *impl_; }
      Interface &impl () { assert( impl_ ); return *impl_; }

      std::unique_ptr< Interface > impl_;
    };

  }
}


#endif // #ifndef ELLIPTC_MODEL_HH
