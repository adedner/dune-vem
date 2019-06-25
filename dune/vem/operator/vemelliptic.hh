#ifndef VEMELLIPTIC_HH
#define VEMELLIPTIC_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/timer.hh>

#include <dune/fem/operator/common/operator.hh>
#include <dune/fem/operator/common/stencil.hh>
#include <dune/fem/quadrature/cachingquadrature.hh>

#include <dune/fem/operator/common/differentiableoperator.hh>
#include <dune/fem/io/parameter.hh>
#include <dune/fem/schemes/elliptic.hh>

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
    //
    typedef Dune::Fem::CachingQuadrature<GridPartType, 0> QuadratureType;
    typedef Dune::Fem::ElementQuadrature<GridPartType, 1> FaceQuadratureType;
    //
  public:
    //! contructor
    VEMEllipticOperator ( const DomainDiscreteFunctionSpaceType &dSpace,
                          const RangeDiscreteFunctionSpaceType &rSpace,
                          ModelType &model,
                          const Dune::Fem::ParameterReader &parameter = Dune::Fem::Parameter::container() )
    : model_( model ),
      dSpace_(dSpace), rSpace_(rSpace),
      baseOperator_(dSpace,rSpace,model,parameter)
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
                          ModelType &model,
                          const Dune::Fem::ParameterReader &parameter = Dune::Fem::Parameter::container() )
    : VEMEllipticOperator( rangeSpace, rangeSpace, model, parameter )
    {}


    //! application operator
    virtual void
      operator()(const DomainDiscreteFunctionType &u,
          RangeDiscreteFunctionType &w) const;

    ModelType &model() const
    { return model_; }
    const DomainDiscreteFunctionSpaceType& domainSpace() const
    { return dSpace_; }
    const RangeDiscreteFunctionSpaceType& rangeSpace() const
    { return rSpace_; }

  private:
    ModelType &model_;
    const DomainDiscreteFunctionSpaceType &dSpace_;
    const RangeDiscreteFunctionSpaceType &rSpace_;
    EllipticOperator<DomainDiscreteFunction,RangeDiscreteFunction,Model> baseOperator_;
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
                     ModelType &model,
                     const Dune::Fem::ParameterReader &parameter = Dune::Fem::Parameter::container() )
    : BaseType( rangeSpace, model )
    , baseOperator_(rangeSpace,model,parameter)
  {}
  DifferentiableVEMEllipticOperator ( const DomainDiscreteFunctionSpaceType &dSpace,
                                      const RangeDiscreteFunctionSpaceType &rSpace,
                                      ModelType &model,
                                      const Dune::Fem::ParameterReader &parameter = Dune::Fem::Parameter::container() )
  : BaseType( dSpace, rSpace, model, parameter )
  , baseOperator_(dSpace,rSpace,model,parameter)
  {}

  //! method to setup the jacobian of the operator for storage in a matrix
  void jacobian(const DomainDiscreteFunctionType &u,
      JacobianOperatorType &jOp) const;

  using BaseType::model;
  DifferentiableEllipticOperator<JacobianOperator,Model> baseOperator_;
};

// Implementation of VEMEllipticOperator
// -------------------------------------

template<class DomainDiscreteFunction, class RangeDiscreteFunction, class Model>
  void VEMEllipticOperator<DomainDiscreteFunction, RangeDiscreteFunction, Model>
  ::operator()(const DomainDiscreteFunctionType &u,
      RangeDiscreteFunctionType &w) const
{
  baseOperator_(u,w);
  if (! std::is_same<DomainDiscreteFunctionSpaceType,RangeDiscreteFunctionSpaceType>::value)
    return;

  // get discrete function space
  const RangeDiscreteFunctionSpaceType &dfSpace = w.space();
  const int blockSize = dfSpace.localBlockSize; // is equal to 1 for scalar functions

  std::vector<RangeRangeType> VectorOfAveragedDiffusionCoefficients (dfSpace.agglomeration().size(), RangeRangeType(0));
  const auto &agIndexSet = dfSpace.blockMapper().indexSet();

  // iterate over grid
  const GridPartType &gridPart = w.gridPart();
  for (const auto &entity : Dune::elements(
        static_cast<typename GridPartType::GridViewType>(gridPart),
        Dune::Partitions::interiorBorder))
  {
    model().init(entity);
    // get elements geometry
    const GeometryType &geometry = entity.geometry();

    const int numVertices = agIndexSet.numPolyVertices(entity, GridPartType::dimension);

    // get local representation of the discrete functions
    const DomainLocalFunctionType uLocal = u.localFunction(entity);
    RangeLocalFunctionType wLocal = w.localFunction(entity);

    auto& refElement = Dune::ReferenceElements<double, 2>::general( entity.type());

    const std::size_t agglomerate = dfSpace.agglomeration().index( entity);
    const auto &bbox = agIndexSet.boundingBox( agglomerate );

    for (const auto &intersection : Dune::intersections(
         static_cast<typename GridPartType::GridViewType>(gridPart), entity))
    {
      if( !intersection.boundary() && (dfSpace.agglomeration().index( intersection.outside() ) == agglomerate) )
        continue;

      const int faceIndex = intersection.indexInInside();
      const int numEdgeVertices = refElement.size(faceIndex, 1, GridPartType::dimension);
      for (int i = 0; i < numEdgeVertices; ++i)
      {
        // local vertex number in the triangle/quad, this is not the local vertex number of the polygon!
        const int j = refElement.subEntity(faceIndex, 1, i, GridPartType::dimension);
        // global coordinate of the vertex
        DomainType globalPoint = geometry.corner(j);
        // local coordinate of the vertex
        DomainType localPoint = geometry.local(globalPoint);

        DomainRangeType vu;
        DomainJacobianRangeType dvu;
        uLocal.evaluate(localPoint, vu);
        uLocal.jacobian(localPoint, dvu);
#if 0
        DomainJacobianRangeType grad1;
        grad1[0][0] = 1.;
        grad1[0][1] = 0.;
        RangeJacobianRangeType a(0);
        model().diffusiveFlux(localPoint, vu, grad1, a);
        RangeRangeType Dcoeff;
        Dcoeff[0] = a[0][0];
#else
        RangeRangeType Dcoeff = model().gradStabilization(localPoint,vu);
        Dcoeff.axpy(bbox.volume(), model().massStabilization(localPoint,vu) );
#endif

        double factor = 1. / (2.0 * numVertices);
        VectorOfAveragedDiffusionCoefficients[agglomerate].axpy(factor,Dcoeff);
      }
    }
  }

  // assemble the stablisation matrix
  std::vector<bool> stabilization(dfSpace.agglomeration().size(), false);
  for (const auto &entity : Dune::elements(
        static_cast<typename GridPartType::GridViewType>(gridPart),
        Dune::Partitions::interiorBorder)) {
    model().init(entity);
    RangeLocalFunctionType wLocal = w.localFunction(entity);
    const DomainLocalFunctionType uLocal = u.localFunction(entity);
    const std::size_t agglomerate = dfSpace.agglomeration().index( entity);
    if (!stabilization[dfSpace.agglomeration().index(entity)])
    {
      const auto &stabMatrix = dfSpace.stabilization(entity);
      assert( stabMatrix.cols()*blockSize == uLocal.size() );
      assert( stabMatrix.rows()*blockSize == wLocal.size() );
      for (std::size_t r = 0; r < stabMatrix.rows(); ++r)
        for (std::size_t c = 0; c < stabMatrix.cols(); ++c)
          for (std::size_t b = 0; b < blockSize; ++b)
            wLocal[r*blockSize+b] +=
              VectorOfAveragedDiffusionCoefficients[agglomerate][0]
                * stabMatrix[r][c] * uLocal[c*blockSize+b];
      stabilization[dfSpace.agglomeration().index(entity)] = true;
    }
  }
  w.communicate();
}

// Implementation of DifferentiableVEMEllipticOperator
// ---------------------------------------------------

template<class JacobianOperator, class Model>
void DifferentiableVEMEllipticOperator<JacobianOperator, Model>
::jacobian( const DomainDiscreteFunctionType &u, JacobianOperator &jOp) const
{
  baseOperator_.jacobian(u,jOp);
  if (! std::is_same<DomainDiscreteFunctionSpaceType,RangeDiscreteFunctionSpaceType>::value)
    return;

  Dune::Timer timer;
  typedef typename JacobianOperator::LocalMatrixType LocalMatrixType;
  typedef typename DomainDiscreteFunctionSpaceType::BasisFunctionSetType DomainBasisFunctionSetType;
  typedef typename RangeDiscreteFunctionSpaceType::BasisFunctionSetType RangeBasisFunctionSetType;

  const DomainDiscreteFunctionSpaceType &domainSpace = jOp.domainSpace();
  const RangeDiscreteFunctionSpaceType &rangeSpace = jOp.rangeSpace();

  // std::cout << "   in assembly: setting up vectors    " << timer.elapsed() << std::endl;;
  const int domainBlockSize = domainSpace.localBlockSize; // is equal to 1 for scalar functions
  // Note the following is much! too large since it assumes e.g. all vertices in one polygon

  std::vector<RangeRangeType> VectorOfAveragedDiffusionCoefficients (rangeSpace.agglomeration().size(), RangeRangeType(0));
  std::vector<RangeRangeType> VectorOfAveragedLinearlisedDiffusionCoefficients (rangeSpace.agglomeration().size(), RangeRangeType(0));

  RangeRangeType Dcoeff(0);
  RangeRangeType LinDcoeff(0);
  const GridPartType &gridPart = rangeSpace.gridPart();

  // std::cout << "   in assembly: computing index set   " << timer.elapsed() << std::endl;;
  const auto &agIndexSet    = rangeSpace.blockMapper().indexSet();
  const auto &agglomeration = rangeSpace.agglomeration();
  // std::cout << "   in assembly: start element loop size=" << rangeSpace.gridPart().grid().size(0) << " time=  " << timer.elapsed() << std::endl;;

  for (const auto &entity : Dune::elements(
        static_cast<typename GridPartType::GridViewType>(gridPart),
        Dune::Partitions::interiorBorder)) {
    const GeometryType &geometry = entity.geometry();
    model().init(entity);
    const DomainLocalFunctionType uLocal = u.localFunction(entity);
    LocalMatrixType jLocal = jOp.localMatrix(entity, entity);

    const int agglomerate = agglomeration.index(entity); // the polygon we are integrating
    const auto &bbox = agIndexSet.boundingBox( agglomerate );
    const int numVertices = agIndexSet.numPolyVertices(entity, GridPartType::dimension);
    // Lines copied from below just before the quadrature loop:
    // For Stabilisation..
    auto& refElement = Dune::ReferenceElements<double, 2>::general( entity.type());

    for (const auto &intersection : Dune::intersections(
          static_cast<typename GridPartType::GridViewType>(gridPart), entity))
    {
      if( !intersection.boundary() && (agglomeration.index( intersection.outside() ) == agglomerate) )
        continue;

      const int faceIndex = intersection.indexInInside();
      const int numEdgeVertices = refElement.size(faceIndex, 1, GridPartType::dimension);
      for (int i = 0; i < numEdgeVertices; ++i)
      {
        const int j = refElement.subEntity(faceIndex, 1, i, GridPartType::dimension);
        // global coordinate of the vertex
        DomainType globalPoint = geometry.corner(j);
        // local coordinate of the vertex
        DomainType localPoint = geometry.local(globalPoint);
        DomainRangeType vu;
        DomainJacobianRangeType dvu;
        uLocal.evaluate(localPoint, vu);
        uLocal.jacobian(localPoint, dvu);

#if 0
        //!!! model().diffusionCoefficient(localPoint, vu, dvu, Dcoeff);
        //!!! model().lindiffusionCoefficient(localPoint, vu, dvu, LinDcoeff);
        //!!! hack for accessing diffusion coefficient:
        DomainJacobianRangeType grad1;
        grad1[0][0] = 1.;
        grad1[0][1] = 0.;
        RangeJacobianRangeType a(0);
        model().diffusiveFlux(localPoint, vu, grad1, a);
        Dcoeff[0] = a[0][0];
        //std::cout << "Dcoeff= " << Dcoeff[0] << std::endl;
        //!!! hack for accessing derivative of diffusion coefficient:
        DomainRangeType v(1.);
        RangeJacobianRangeType da(0);
        DomainJacobianRangeType gradbar1;
        DomainJacobianRangeType dval;
        gradbar1[0][0] = 1.;
        gradbar1[0][1] = 0.;
        dval[0][0] = 0;
        dval[0][1] = 0;
        model().linDiffusiveFlux(vu, gradbar1, localPoint, v, dval, da);
        LinDcoeff[0] = da[0][0];
        //std::cout << "LinDcoeff= " << LinDcoeff[0] << std::endl;
        // Each vertex visited twice in looping around the edges, so we divide by 2
#else
        RangeRangeType Dcoeff = model().gradStabilization(localPoint,vu);
        Dcoeff.axpy(bbox.volume(), model().massStabilization(localPoint,vu) );
        RangeRangeType LinDcoeff = model().linGradStabilization(localPoint,vu);
        LinDcoeff.axpy(bbox.volume(), model().linMassStabilization(localPoint,vu) );
#endif

        double factor = 1./(2. * numVertices);
        VectorOfAveragedDiffusionCoefficients[agglomerate].axpy(factor,Dcoeff);
        VectorOfAveragedLinearlisedDiffusionCoefficients[agglomerate].axpy(factor,LinDcoeff);
      }
    }
  } // element loop end

  std::vector<bool> stabilization(agglomeration.size(), false);
  for (const auto &entity : Dune::elements(
        static_cast<typename GridPartType::GridViewType>(gridPart),
        Dune::Partitions::interiorBorder))
  {
    const std::size_t agglomerate = agglomeration.index( entity);
    if (!stabilization[agglomerate])
    {
      const auto &stabMatrix = rangeSpace.stabilization(entity);
      LocalMatrixType jLocal = jOp.localMatrix(entity, entity);
      assert( jLocal.rows()    == stabMatrix.rows()*domainBlockSize );
      assert( jLocal.columns() == stabMatrix.cols()*domainBlockSize );
      for (std::size_t r = 0; r < stabMatrix.rows(); ++r)
        for (std::size_t c = 0; c < stabMatrix.cols(); ++c)
          for (std::size_t b = 0; b < domainBlockSize; ++b)
            jLocal.add(r*domainBlockSize+b, c*domainBlockSize+b, VectorOfAveragedDiffusionCoefficients[agglomerate][0]  * stabMatrix[r][c]);
      const int nE = agIndexSet.numPolyVertices(entity, GridPartType::dimension);
      const DomainLocalFunctionType uLocal = u.localFunction(entity);
      for (std::size_t c = 0; c < stabMatrix.cols(); ++c)
        for (std::size_t r = 0; r < stabMatrix.rows(); ++r)
          for (std::size_t ccc = 0; ccc < stabMatrix.cols(); ++ccc)
            for (std::size_t b = 0; b < domainBlockSize; ++b)
              jLocal.add(r*domainBlockSize+b, c*domainBlockSize+b, VectorOfAveragedLinearlisedDiffusionCoefficients[agglomerate][0]  * stabMatrix[r][ccc] * uLocal[ccc] / (nE));
      stabilization[agglomerate] = true;
    }
  }
  jOp.communicate();
  // std::cout << "   in assembly: final    " << timer.elapsed() << std::endl;;
}
#endif // #ifndef VEMELLIPTIC_HH
