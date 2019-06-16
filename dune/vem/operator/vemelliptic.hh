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
    VEMEllipticOperator ( const RangeDiscreteFunctionSpaceType &rangeSpace,
                          ModelType &model,
                          const Dune::Fem::ParameterReader &parameter = Dune::Fem::Parameter::container() )
    : model_( model ),
      dSpace_(rangeSpace), rSpace_(rangeSpace),
      baseOperator_(rangeSpace,model,parameter)
    {}
    VEMEllipticOperator ( const DomainDiscreteFunctionSpaceType &dSpace,
                          const RangeDiscreteFunctionSpaceType &rSpace,
                          ModelType &model,
                          const Dune::Fem::ParameterReader &parameter = Dune::Fem::Parameter::container() )
    : model_( model ),
      dSpace_(dSpace), rSpace_(rSpace),
      baseOperator_(dSpace,rSpace,model,parameter)
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

  // get discrete function space
  const RangeDiscreteFunctionSpaceType &dfSpace = w.space();
  //
  std::vector<bool> stabilization(dfSpace.agglomeration().size(), false);
  //
  std::vector<double> polygonareas(dfSpace.agglomeration().size(), 0.0); // added 3rd June 2019 for source term (GD)
  std::vector<RangeRangeType> VectorOfAveragedDiffusionCoefficients (dfSpace.agglomeration().size(), RangeRangeType(0));
  std::vector<RangeRangeType> VectorOfAveragedSourceCoefficients (dfSpace.agglomeration().size(), RangeRangeType(0));
  const auto &agIndexSet = dfSpace.blockMapper().indexSet();
  // iterate over grid
  const GridPartType &gridPart = w.gridPart();
  for (const auto &entity : Dune::elements(
        static_cast<typename GridPartType::GridViewType>(gridPart),
        Dune::Partitions::interiorBorder)) {
    model().init(entity);
    // get elements geometry
    const GeometryType &geometry = entity.geometry();
    //
    const int numVertices = agIndexSet.numPolyVertices(entity,
        GridPartType::dimension);

    // get local representation of the discrete functions
    const DomainLocalFunctionType uLocal = u.localFunction(entity);
    RangeLocalFunctionType wLocal = w.localFunction(entity);
    //
    auto& refElement = Dune::ReferenceElements<double, 2>::general(
        entity.type());

    RangeRangeType Dcoeff(0);
    const std::size_t agglomerate = dfSpace.agglomeration().index(
        entity);

    // ====
    for (const auto &intersection : Dune::intersections(
          static_cast<typename GridPartType::GridViewType>(gridPart), entity))
    {
      if( !intersection.boundary() && (dfSpace.agglomeration().index( intersection.outside() ) == agglomerate) )
        continue;
      //
      const int faceIndex = intersection.indexInInside();
      const int numEdgeVertices = refElement.size(faceIndex, 1, GridPartType::dimension);
      for (int i = 0; i < numEdgeVertices; ++i)
      {
        // local vertex number in the triangle/quad, this is not the local vertex number
        // of the polygon!
        const int j = refElement.subEntity(faceIndex, 1, i, GridPartType::dimension);
        // global coordinate of the vertex
        DomainType GlobalPoint = geometry.corner(j);
        // local coordinate of the vertex
        DomainType LocalPoint = geometry.local(GlobalPoint);

        DomainRangeType vu;
        DomainJacobianRangeType dvu;
        uLocal.evaluate(LocalPoint, vu);
        uLocal.jacobian(LocalPoint, dvu);

        //!!!! model().diffusionCoefficient( LocalPoint, vu, dvu, Dcoeff);
        DomainJacobianRangeType grad1;
        grad1[0][0] = 1.;
        grad1[0][1] = 0.;
        RangeJacobianRangeType a(0);
        model().diffusiveFlux(GlobalPoint, vu, grad1, a);
        Dcoeff[0] = a[0][0];
        //!!!! 30.05.2019- Ganesh
        //!!!! We need similar mechanism for source term as well.
        //!!!! No need to pass entity anymore.
        //!!!! source(const Entity &entity, const Point &x, const RangeType &value,
        //!!!!        const JacobianRangeType &gradient, JacobianRangeType &flux ) const
        RangeRangeType mcoeff(0);
        model().source(GlobalPoint, vu, grad1, mcoeff );
        double factor = 1. / (2.0 * numVertices);
        VectorOfAveragedDiffusionCoefficients[agglomerate].axpy(factor,Dcoeff);
        //!!!! define a vector of avg source coeff somewhere..
        VectorOfAveragedSourceCoefficients[agglomerate].axpy(factor,mcoeff);
      }
    }
    polygonareas[agglomerate] += geometry.volume();
  }
  // assemble the stablisation matrix
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
      for (std::size_t r = 0; r < stabMatrix.rows(); ++r)
        for (std::size_t c = 0; c < stabMatrix.cols(); ++c)
          wLocal[r] += ( VectorOfAveragedDiffusionCoefficients[agglomerate][0]  + VectorOfAveragedSourceCoefficients[agglomerate][0] * polygonareas[agglomerate] ) * stabMatrix[r][c] * uLocal[c];
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
  // std::cout << "starting assembly\n";

  Dune::Timer timer;
  typedef typename JacobianOperator::LocalMatrixType LocalMatrixType;
  typedef typename DomainDiscreteFunctionSpaceType::BasisFunctionSetType DomainBasisFunctionSetType;
  typedef typename RangeDiscreteFunctionSpaceType::BasisFunctionSetType RangeBasisFunctionSetType;

  const DomainDiscreteFunctionSpaceType &domainSpace = jOp.domainSpace();
  const RangeDiscreteFunctionSpaceType &rangeSpace = jOp.rangeSpace();

  // std::cout << "   in assembly: setting up vectors    " << timer.elapsed() << std::endl;;
  const int domainBlockSize = domainSpace.localBlockSize; // is equal to 1 for scalar functions
  // Note the following is much! too large since it assumes e.g. all vertices in one polygon

  std::vector<double> polygonareas(rangeSpace.agglomeration().size(), 0.0);
  std::vector<RangeRangeType> VectorOfAveragedDiffusionCoefficients (rangeSpace.agglomeration().size(), RangeRangeType(0));
  std::vector<RangeRangeType> VectorOfAveragedLinearlisedDiffusionCoefficients (rangeSpace.agglomeration().size(), RangeRangeType(0));
  std::vector<RangeRangeType> VectorOfAveragedmCoefficients(rangeSpace.agglomeration().size(), RangeRangeType(0));
  std::vector<RangeRangeType> VectorOfAveragedLinearlisedmCoefficients(rangeSpace.agglomeration().size(), RangeRangeType(0));
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
        DomainType GlobalPoint = geometry.corner(j);
        // local coordinate of the vertex
        DomainType LocalPoint = geometry.local(GlobalPoint);
        DomainRangeType vu;
        DomainJacobianRangeType dvu;
        uLocal.evaluate(LocalPoint, vu);
        uLocal.jacobian(LocalPoint, dvu);

        //!!! model().diffusionCoefficient(LocalPoint, vu, dvu, Dcoeff);
        //!!! model().lindiffusionCoefficient(LocalPoint, vu, dvu, LinDcoeff);
        //!!! hack for accessing diffusion coefficient:
        DomainJacobianRangeType grad1;
        grad1[0][0] = 1.;
        grad1[0][1] = 0.;
        RangeJacobianRangeType a(0);
        model().diffusiveFlux(GlobalPoint, vu, grad1, a);
        Dcoeff[0] = a[0][0];
        //!!! hack for accessing derivative of diffusion coefficient:
        DomainRangeType v(1.);
        DomainJacobianRangeType da(0);
        DomainJacobianRangeType gradbar1;
        DomainJacobianRangeType dval;
        gradbar1[0][0] = 1.;
        gradbar1[0][1] = 0.;
        dval[0][0] = 0;
        dval[0][1] = 0;
        model().linDiffusiveFlux(vu, gradbar1, GlobalPoint, v, dval, da);
        LinDcoeff[0] = da[0][0];
        //!!!! hack for accessing derivative of mass term: Ganesh, 30.05.2019
        // if mass terms or right hand side is present
        //model().linSource( u0, jacU0, entity, quadrature[ pt ], phi[ localCol ], dphi[localCol], aphi );
        RangeRangeType mcoeff(0);
        model().source(GlobalPoint, vu, grad1, mcoeff );
        RangeRangeType Linmcoeff(0);
        model().linSource( vu, gradbar1, GlobalPoint, v, dval, Linmcoeff );
        std::cout << "mcoeff= " << mcoeff[0] << std::endl;
        std::cout << "Linmcoeff= " << Linmcoeff[0] << std::endl;
        // Each vertex visited twice in looping around the edges, so we divide by 2
        double factor = 1./(2. * numVertices);
        VectorOfAveragedDiffusionCoefficients[agglomerate].axpy(factor,Dcoeff);
        VectorOfAveragedLinearlisedDiffusionCoefficients[agglomerate].axpy(factor,LinDcoeff);
        // vectors for the averaged coefficients of mass term
        VectorOfAveragedmCoefficients[agglomerate].axpy(factor,mcoeff);
        VectorOfAveragedLinearlisedmCoefficients[agglomerate].axpy(factor,Linmcoeff);
      }
    }
    polygonareas[agglomerate] += geometry.volume();
  } // element loop end

  // the second element loop to add the stabilisation term
  // std::cout << "   in assembly: second element loop    " << timer.elapsed() << std::endl;;

  std::vector<double> localmeshsizevec(rangeSpace.agglomeration().size(),0.0);
  for (int i = 0; i < rangeSpace.agglomeration().size(); ++i) {
    localmeshsizevec[i] = std::pow(polygonareas[i],
        1.0 / GridPartType::dimension);
  }
  double averagemeshsize = 0.0;
  for (int i = 0; i < rangeSpace.agglomeration().size(); ++i)
    averagemeshsize += localmeshsizevec[i]
      / rangeSpace.agglomeration().size();

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
      for (std::size_t r = 0; r < stabMatrix.rows(); ++r)
        for (std::size_t c = 0; c < stabMatrix.cols(); ++c)
          jLocal.add(r, c, ( VectorOfAveragedDiffusionCoefficients[agglomerate][0]  + VectorOfAveragedmCoefficients[agglomerate][0] * polygonareas[agglomerate] ) * stabMatrix[r][c]);
      const int nE = agIndexSet.numPolyVertices(entity, GridPartType::dimension);
      const DomainLocalFunctionType uLocal = u.localFunction(entity);
      for (std::size_t c = 0; c < stabMatrix.cols(); ++c)
        for (std::size_t r = 0; r < stabMatrix.rows(); ++r)
          for (std::size_t ccc = 0; ccc < stabMatrix.cols(); ++ccc)
            jLocal.add(r, c, ( VectorOfAveragedLinearlisedDiffusionCoefficients [agglomerate][0] + VectorOfAveragedLinearlisedmCoefficients[agglomerate][0] * polygonareas[agglomerate] ) * stabMatrix[r][ccc] * uLocal[ccc] / (nE));
      stabilization[agglomerate] = true;
    }
  }
  jOp.communicate();
  std::cout << "   in assembly: final    " << timer.elapsed() << std::endl;;
}
#endif // #ifndef VEMELLIPTIC_HH
