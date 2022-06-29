#ifndef VEMELLIPTIC_HH
#define VEMELLIPTIC_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/timer.hh>

#include <dune/fem/operator/common/operator.hh>
#include <dune/fem/operator/common/stencil.hh>
#include <dune/fem/quadrature/cachingquadrature.hh>
#include <dune/fem/function/localfunction/const.hh>
#include <dune/fem/function/common/localcontribution.hh>
#include <dune/fem/common/bindguard.hh>

#include <dune/fem/operator/common/differentiableoperator.hh>
#include <dune/fem/io/parameter.hh>
#include <dune/fem/schemes/elliptic.hh>
#include <dune/fem/schemes/galerkin.hh>

#include <dune/vem/agglomeration/indexset.hh>

// VEMEllipticOperator
// -------------------
template<class DomainDiscreteFunction, class RangeDiscreteFunction, class Model>
  struct VEMEllipticOperator: public virtual Dune::Fem::Operator<
                              DomainDiscreteFunction, RangeDiscreteFunction>
{
  public:
    typedef DomainDiscreteFunction DomainDiscreteFunctionType;
    typedef RangeDiscreteFunction RangeDiscreteFunctionType;
    typedef Model ModelType;
    typedef Model                  DirichletModelType;
    //
    typedef typename DomainDiscreteFunctionType::DiscreteFunctionSpaceType DomainDiscreteFunctionSpaceType;
    typedef Dune::Fem::ConstLocalFunction< DomainDiscreteFunction > DomainLocalFunctionType;
    typedef typename DomainLocalFunctionType::RangeType DomainRangeType;
    typedef typename DomainLocalFunctionType::JacobianRangeType DomainJacobianRangeType;
    typedef typename RangeDiscreteFunctionType::DiscreteFunctionSpaceType RangeDiscreteFunctionSpaceType;
    typedef Dune::Fem::AddLocalContribution< RangeDiscreteFunctionType > RangeLocalContributionType;
    typedef RangeLocalContributionType  RangeLocalFunctionType;
    typedef typename RangeLocalFunctionType::RangeType RangeRangeType;
    typedef typename RangeLocalFunctionType::JacobianRangeType RangeJacobianRangeType;
    // the following types must be identical for domain and range
    typedef typename RangeDiscreteFunctionSpaceType::IteratorType IteratorType;
    typedef typename IteratorType::Entity EntityType;
    typedef typename EntityType::Geometry GeometryType;
    typedef typename RangeDiscreteFunctionSpaceType::DomainType DomainType;
    typedef typename RangeDiscreteFunctionSpaceType::GridPartType GridPartType;
    typedef typename GridPartType::IntersectionIteratorType IntersectionIteratorType;
    typedef typename IntersectionIteratorType::Intersection IntersectionType;
    //
    typedef Dune::Fem::CachingQuadrature<GridPartType, 0> QuadratureType;
    typedef Dune::Fem::ElementQuadrature<GridPartType, 1> FaceQuadratureType;
    //
  public:
    //! contructor
    VEMEllipticOperator ( const DomainDiscreteFunctionSpaceType &dSpace,
                          const RangeDiscreteFunctionSpaceType &rSpace,
                          ModelType model,
                          const Dune::Fem::ParameterReader &parameter = Dune::Fem::Parameter::container() )
    : dSpace_(dSpace), rSpace_(rSpace),
      // baseOperator_(dSpace,rSpace,model,parameter)
      baseOperator_(dSpace.gridPart(),model)
    {
#if 0
      std::size_t aSize = rSpace.agglomeration().size();
      for( std::size_t agglomerate = 0; agglomerate < aSize; ++agglomerate )
      {
        const std::size_t dNumDofs = dSpace.blockMapper().numDofs( agglomerate );
        const std::size_t rNumDofs = rSpace.blockMapper().numDofs( agglomerate );
        Stabilization &stabilization = stabilizations_[ agglomerate ];
        stabilization.resize( rNumDofs, dNumDofs, 0 );
        for( std::size_t i = 0; i < rNumDofs; ++i )
          for( std::size_t j = 0; j < dNumDofs; ++j )
            for( std::size_t k = 0; k < numDofs; ++k )
              stabilization[ i ][ j ] += S[ k ][ i ] * S[ k ][ j ];
      }
#endif
    }
    VEMEllipticOperator ( const RangeDiscreteFunctionSpaceType &rangeSpace,
                          ModelType model,
                          const Dune::Fem::ParameterReader &parameter = Dune::Fem::Parameter::container() )
    : VEMEllipticOperator( rangeSpace, rangeSpace, model, parameter )
    {}


    //! application operator
    virtual void
      operator()(const DomainDiscreteFunctionType &u,
          RangeDiscreteFunctionType &w) const;

    ModelType &model() const
    { return baseOperator_.model(); }
    const DomainDiscreteFunctionSpaceType& domainSpace() const
    { return dSpace_; }
    const RangeDiscreteFunctionSpaceType& rangeSpace() const
    { return rSpace_; }
    void setQuadratureOrders(unsigned int interior, unsigned int surface)
    { baseOperator_.setQuadratureOrders(interior,surface); }

  private:
    const DomainDiscreteFunctionSpaceType &dSpace_;
    const RangeDiscreteFunctionSpaceType &rSpace_;
    // EllipticOperator<DomainDiscreteFunction,RangeDiscreteFunction,Model> baseOperator_;
    Dune::Fem::GalerkinOperator<Model,DomainDiscreteFunction,RangeDiscreteFunction> baseOperator_;
};

// DifferentiableVEMEllipticOperator
// ------------------------------
//! [Class for linearizable elliptic operator]
template<class JacobianOperator, class Model>
  struct DifferentiableVEMEllipticOperator: public VEMEllipticOperator<
                                            typename JacobianOperator::DomainFunctionType,
                                            typename JacobianOperator::RangeFunctionType, Model>,
                                            public Dune::Fem::DifferentiableOperator<JacobianOperator>
                                            //! [Class for linearizable VEMelliptic operator]
{
  public:
  typedef VEMEllipticOperator<typename JacobianOperator::DomainFunctionType,
  typename JacobianOperator::RangeFunctionType, Model> BaseType;

  typedef JacobianOperator JacobianOperatorType;

  typedef typename BaseType::DomainDiscreteFunctionType DomainDiscreteFunctionType;
  typedef typename BaseType::RangeDiscreteFunctionType RangeDiscreteFunctionType;
  typedef typename BaseType::ModelType ModelType;

  typedef typename DomainDiscreteFunctionType::DiscreteFunctionSpaceType DomainDiscreteFunctionSpaceType;
  typedef typename DomainDiscreteFunctionType::LocalFunctionType DomainLocalFunctionType;
  typedef typename DomainLocalFunctionType::RangeType DomainRangeType;
  typedef typename DomainLocalFunctionType::JacobianRangeType DomainJacobianRangeType;
  typedef typename RangeDiscreteFunctionType::DiscreteFunctionSpaceType RangeDiscreteFunctionSpaceType;
  typedef typename RangeDiscreteFunctionType::LocalFunctionType RangeLocalFunctionType;
  typedef typename RangeLocalFunctionType::RangeType RangeRangeType;
  typedef typename RangeLocalFunctionType::JacobianRangeType RangeJacobianRangeType;

  // the following types must be identical for domain and range
  typedef typename RangeDiscreteFunctionSpaceType::IteratorType IteratorType;
  typedef typename IteratorType::Entity EntityType;
  typedef typename EntityType::Geometry GeometryType;
  typedef typename RangeDiscreteFunctionSpaceType::DomainType DomainType;
  typedef typename RangeDiscreteFunctionSpaceType::GridPartType GridPartType;
  typedef typename GridPartType::IntersectionIteratorType IntersectionIteratorType;
  typedef typename IntersectionIteratorType::Intersection IntersectionType;

  typedef typename BaseType::QuadratureType QuadratureType;
  // quadrature for faces - used for Neuman b.c.
  typedef typename BaseType::FaceQuadratureType FaceQuadratureType;

  public:
  //! contructor
  DifferentiableVEMEllipticOperator ( const RangeDiscreteFunctionSpaceType &rangeSpace,
                     ModelType model,
                     const Dune::Fem::ParameterReader &parameter = Dune::Fem::Parameter::container() )
    : BaseType( rangeSpace, model )
    , baseOperator_(rangeSpace,model)
  {}
  DifferentiableVEMEllipticOperator ( const DomainDiscreteFunctionSpaceType &dSpace,
                                      const RangeDiscreteFunctionSpaceType &rSpace,
                                      ModelType model,
                                      const Dune::Fem::ParameterReader &parameter = Dune::Fem::Parameter::container() )
  : BaseType( dSpace, rSpace,  model, parameter )
  // , baseOperator_(dSpace,rSpace,true, model,parameter)
  , baseOperator_(dSpace,rSpace,model)
  {}

  //! method to setup the jacobian of the operator for storage in a matrix
  void jacobian(const DomainDiscreteFunctionType &u,
      JacobianOperatorType &jOp) const;

  using BaseType::model;
  // DifferentiableEllipticOperator<JacobianOperator,Model> baseOperator_;
  Dune::Fem::DifferentiableGalerkinOperator<Model,JacobianOperator> baseOperator_;
};

// Implementation of VEMEllipticOperator
// -------------------------------------

template<class DomainDiscreteFunction, class RangeDiscreteFunction, class Model>
  void VEMEllipticOperator<DomainDiscreteFunction, RangeDiscreteFunction, Model>
  ::operator()(const DomainDiscreteFunctionType &u,
      RangeDiscreteFunctionType &w) const
{
  baseOperator_(u,w);
#if 1
  if (! std::is_same<DomainDiscreteFunctionSpaceType,RangeDiscreteFunctionSpaceType>::value)
    return;

  // get discrete function space
  const RangeDiscreteFunctionSpaceType &dfSpace = w.space();
  const int blockSize = dfSpace.localBlockSize; // is equal to 1 for scalar functions

  std::vector<RangeRangeType> VectorOfAveragedDiffusionCoefficients (dfSpace.agglomeration().size(), RangeRangeType(0));
  const auto &agIndexSet = dfSpace.blockMapper().indexSet();

  DomainLocalFunctionType uLocal( u );

  // iterate over grid
  const GridPartType &gridPart = w.gridPart();
  for (const auto &entity : Dune::elements( gridPart, Dune::Partitions::interiorBorder ) )
  {
    model().init(entity);
    // get elements geometry
    const GeometryType &geometry = entity.geometry();

    const int numVertices = agIndexSet.numPolyVertices(entity, GridPartType::dimension);

    // get local representation of the discrete functions
    auto uGuard = Dune::Fem::bindGuard( uLocal, entity );

    auto& refElement = Dune::ReferenceElements<double, 2>::general( entity.type());

    const std::size_t agglomerate = dfSpace.agglomeration().index( entity);
    const auto &bbox = agIndexSet.boundingBox( agglomerate );

    for (const auto &intersection : Dune::intersections( gridPart, entity ))
    {
      if( !intersection.boundary() && (dfSpace.agglomeration().index( intersection.outside() ) == agglomerate) )
        continue;

      const int faceIndex = intersection.indexInInside();
      const int numEdgeVertices = refElement.size(faceIndex, 1, GridPartType::dimension);
      double factor = 1. / double(numEdgeVertices * numVertices);
      for (int i = 0; i < numEdgeVertices; ++i)
      {
        // local vertex number in the triangle/quad, this is not the local vertex number of the polygon!
        const int j = refElement.subEntity(faceIndex, 1, i, GridPartType::dimension);
        // global coordinate of the vertex
        DomainType globalPoint = geometry.corner(j);
        // local coordinate of the vertex
        DomainType localPoint = geometry.local(globalPoint);

        DomainRangeType vu;
        uLocal.evaluate(localPoint, vu);

        double bbH2 = pow(bbox.volume()/bbox.diameter(),2);
        // std::cout << "vemelliptic:" << pow(bbox.diameter(),2) << " " << bbox.volume() << std::endl;
        RangeRangeType Dcoeff = model().gradStabilization(localPoint,vu);
        Dcoeff.axpy(bbH2, model().massStabilization(localPoint,vu) );
        Dcoeff.axpy(1./bbH2, model().hessStabilization(localPoint,vu) );

        VectorOfAveragedDiffusionCoefficients[agglomerate].axpy(factor,Dcoeff);
      }
    }
  }

  RangeLocalContributionType wLocal( w );

  // assemble the stablisation matrix
  std::vector<bool> stabilization(dfSpace.agglomeration().size(), false);
  for (const auto &entity : Dune::elements( gridPart, Dune::Partitions::interiorBorder))
  {
    auto wGuard = Dune::Fem::bindGuard( wLocal, entity );
    wLocal.clear();

    auto uGuard = Dune::Fem::bindGuard( uLocal, entity );
    const std::size_t agglomerate = dfSpace.agglomeration().index( entity);
    if (!stabilization[dfSpace.agglomeration().index(entity)])
    {
      const auto &stabMatrix = dfSpace.stabilization(entity);
      std::size_t bs = 1;
      if ( stabMatrix.cols()*blockSize == uLocal.size() )
        bs = blockSize;
      assert( stabMatrix.cols()*bs == uLocal.size() );
      assert( stabMatrix.rows()*bs == wLocal.size() );
      // assert( bs == 1 );
      for (std::size_t r = 0; r < stabMatrix.rows(); ++r)
        for (std::size_t c = 0; c < stabMatrix.cols(); ++c)
          for (std::size_t b = 0; b < bs; ++b)
            wLocal[r*bs+b] +=
            VectorOfAveragedDiffusionCoefficients[agglomerate][0] // FIX ME: how to make the coeff. depend on the range dimension
                * stabMatrix[r][c] * uLocal[c*bs+b];
      stabilization[dfSpace.agglomeration().index(entity)] = true;
    }
  }
  w.communicate();
#endif
}

// Implementation of DifferentiableVEMEllipticOperator
// ---------------------------------------------------

template<class JacobianOperator, class Model>
void DifferentiableVEMEllipticOperator<JacobianOperator, Model>
::jacobian( const DomainDiscreteFunctionType &u, JacobianOperator &jOp) const
{
  Dune::Timer timer;
  baseOperator_.jacobian(u,jOp);
#if 1
  if (! std::is_same<DomainDiscreteFunctionSpaceType,RangeDiscreteFunctionSpaceType>::value)
    return;
  // std::cout << "   in assembly: base operator    " << timer.elapsed() << std::endl;

  typedef typename JacobianOperator::LocalMatrixType LocalMatrixType;
  typedef typename DomainDiscreteFunctionSpaceType::BasisFunctionSetType DomainBasisFunctionSetType;
  typedef typename RangeDiscreteFunctionSpaceType::BasisFunctionSetType RangeBasisFunctionSetType;

  const DomainDiscreteFunctionSpaceType &domainSpace = jOp.domainSpace();
  const RangeDiscreteFunctionSpaceType &rangeSpace = jOp.rangeSpace();

  const int domainBlockSize = domainSpace.localBlockSize; // is equal to 1 for scalar functions
  // Note the following is much! too large since it assumes e.g. all vertices in one polygon

  std::vector<RangeRangeType> VectorOfAveragedDiffusionCoefficients (rangeSpace.agglomeration().size(), RangeRangeType(0));
  std::vector<RangeRangeType> VectorOfAveragedLinearlisedDiffusionCoefficients (rangeSpace.agglomeration().size(), RangeRangeType(0));

  RangeRangeType Dcoeff(0);
  RangeRangeType LinDcoeff(0);
  const GridPartType &gridPart = rangeSpace.gridPart();

  const auto &agIndexSet    = rangeSpace.blockMapper().indexSet();
  const auto &agglomeration = rangeSpace.agglomeration();

  typedef typename GridPartType::template Codim< 0 >::EntitySeedType ElementSeedType;
  std::vector< ElementSeedType > stabilization( agglomeration.size() );

  // std::cout << "   in assembly: start element loop time=  " << timer.elapsed() << std::endl;

  DomainLocalFunctionType uLocal( u );

  for (const auto &entity : Dune::elements(gridPart, Dune::Partitions::interiorBorder))
  {
    const GeometryType &geometry = entity.geometry();
    model().init(entity);

    auto uGuard = Dune::Fem::bindGuard( uLocal, entity );

    const unsigned int agglomerate = agglomeration.index(entity); // the polygon we are integrating
    const auto &bbox = agIndexSet.boundingBox( agglomerate );
    const int numVertices = agIndexSet.numPolyVertices(entity, GridPartType::dimension);
    stabilization[ agglomerate ] = entity.seed();

    // Lines copied from below just before the quadrature loop:
    // For Stabilisation..
    auto& refElement = Dune::ReferenceElements<double, 2>::general( entity.type());

    for (const auto &intersection : Dune::intersections(gridPart, entity))
    {
      if( !intersection.boundary() && (agglomeration.index( intersection.outside() ) == agglomerate) )
        continue;

      const int faceIndex = intersection.indexInInside();
      const int numEdgeVertices = refElement.size(faceIndex, 1, GridPartType::dimension);
      double factor = 1./double(numEdgeVertices * numVertices);
      for (int i = 0; i < numEdgeVertices; ++i)
      {
        const int j = refElement.subEntity(faceIndex, 1, i, GridPartType::dimension);
        // global coordinate of the vertex
        DomainType globalPoint = geometry.corner(j);
        // local coordinate of the vertex
        DomainType localPoint = geometry.local(globalPoint);
        DomainRangeType vu;
        uLocal.evaluate(localPoint, vu);

        double bbH2 = pow(bbox.volume()/bbox.diameter(),2);
        // std::cout << "vemelliptic:" << pow(bbox.diameter(),2) << " " << bbox.volume() << std::endl;
        RangeRangeType Dcoeff = model().gradStabilization(localPoint,vu);
        Dcoeff.axpy(bbH2, model().massStabilization(localPoint,vu) );
        Dcoeff.axpy(1./bbH2, model().hessStabilization(localPoint,vu) );
        RangeRangeType LinDcoeff = model().linGradStabilization(localPoint,vu);
        LinDcoeff.axpy(bbH2, model().linMassStabilization(localPoint,vu) );
        LinDcoeff.axpy(1./bbH2, model().linHessStabilization(localPoint,vu) );

        VectorOfAveragedDiffusionCoefficients[agglomerate].axpy(factor,Dcoeff);
        VectorOfAveragedLinearlisedDiffusionCoefficients[agglomerate].axpy(factor,LinDcoeff);
      }
    }
  } // element loop end

  // std::cout << "   in assembly: finished element loop time=  " << timer.elapsed() << std::endl;

  typedef Dune::Fem::TemporaryLocalMatrix< DomainDiscreteFunctionSpaceType,
                                           RangeDiscreteFunctionSpaceType > TemporaryLocalMatrixType;
  TemporaryLocalMatrixType jLocal( domainSpace, rangeSpace );

  for (const auto &seed : stabilization)
  {
    const auto entity = gridPart.entity( seed );
    const std::size_t agglomerate = agglomeration.index( entity );

    const auto &stabMatrix = rangeSpace.stabilization(entity);
    jLocal.init( entity, entity );
    jLocal.clear();

    //??? const int nE = agIndexSet.numPolyVertices(entity, GridPartType::dimension);
    auto uGuard = Dune::Fem::bindGuard( uLocal, entity );
    std::size_t bs = 1;
    if ( stabMatrix.cols()*domainBlockSize == uLocal.size() )
      bs = domainBlockSize;
    // assert( bs == 1);
    assert( jLocal.rows()    == stabMatrix.rows()*bs );
    assert( jLocal.columns() == stabMatrix.cols()*bs );
    assert( stabMatrix.cols()*bs == uLocal.size() );

    for (std::size_t r = 0; r < stabMatrix.rows(); ++r)
      for (std::size_t c = 0; c < stabMatrix.cols(); ++c)
        for (std::size_t b = 0; b < bs; ++b)
        {
          double add = 0;
          for (std::size_t ccc = 0; ccc < stabMatrix.cols(); ++ccc)
            for (std::size_t b = 0; b < bs; ++b)
              add += stabMatrix[r][ccc] * uLocal[ccc*bs+b]; //???  / (nE));
          jLocal.add(r*bs+b, c*bs+b,
                     VectorOfAveragedDiffusionCoefficients[agglomerate][0] * stabMatrix[r][c] +
                     VectorOfAveragedLinearlisedDiffusionCoefficients[agglomerate][0] * add ); // FIX ME: make coeff depend on range * dimension
        }

    jOp.addLocalMatrix( entity, entity, jLocal );
#if 0
    for (std::size_t r = 0; r < stabMatrix.rows(); ++r)
    {
      for (std::size_t c = 0; c < stabMatrix.cols(); ++c)
        for (std::size_t b = 0; b < domainBlockSize; ++b)
          std::cout << r << " " << c << " : "
            << jLocal.get(r*domainBlockSize+b,c*domainBlockSize+b) << "    ";
      std::cout << std::endl;
    }
#endif
  }
  // std::cout << "   in assembly: end stabilization    " << timer.elapsed() << std::endl;
  jOp.flushAssembly();
  // std::cout << "   in assembly: final    " << timer.elapsed() << std::endl;
#endif
}
#endif // #ifndef VEMELLIPTIC_HH
