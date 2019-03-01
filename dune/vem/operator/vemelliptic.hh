#ifndef VEMELLIPTIC_HH
#define VEMELLIPTIC_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/timer.hh>

#include <dune/fem/operator/common/operator.hh>
#include <dune/fem/operator/common/stencil.hh>
#include <dune/fem/quadrature/cachingquadrature.hh>

#include <dune/fem/operator/common/differentiableoperator.hh>
#include <dune/fem/io/parameter.hh>

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
      dSpace_(rangeSpace), rSpace_(rangeSpace)
    {}
    VEMEllipticOperator ( const DomainDiscreteFunctionSpaceType &dSpace,
                          const RangeDiscreteFunctionSpaceType &rSpace,
                          ModelType &model,
                          const Dune::Fem::ParameterReader &parameter = Dune::Fem::Parameter::container() )
    : model_( model ),
      dSpace_(dSpace), rSpace_(rSpace)
      // interiorOrder_(-1), surfaceOrder_(-1)
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
  {}
  DifferentiableVEMEllipticOperator ( const DomainDiscreteFunctionSpaceType &dSpace,
                                      const RangeDiscreteFunctionSpaceType &rSpace,
                                      ModelType &model,
                                      const Dune::Fem::ParameterReader &parameter = Dune::Fem::Parameter::container() )
  : BaseType( dSpace, rSpace, model, parameter )
  {}

  //! method to setup the jacobian of the operator for storage in a matrix
  void jacobian(const DomainDiscreteFunctionType &u,
      JacobianOperatorType &jOp) const;

  using BaseType::model;
};

// Implementation of VEMEllipticOperator
// -------------------------------------

template<class DomainDiscreteFunction, class RangeDiscreteFunction, class Model>
  void VEMEllipticOperator<DomainDiscreteFunction, RangeDiscreteFunction, Model>
  ::operator()(const DomainDiscreteFunctionType &u,
      RangeDiscreteFunctionType &w) const
{
  w.clear();
  // get discrete function space
  const RangeDiscreteFunctionSpaceType &dfSpace = w.space();
  //
  std::vector<bool> stabilization(dfSpace.agglomeration().size(), false);
  //

  std::vector<RangeRangeType> VectorOfAveragedDiffusionCoefficients (dfSpace.agglomeration().size(), RangeRangeType(0));
  // const Dune::Vem::AgglomerationIndexSet<GridPartType> agIndexSet(
  //     dfSpace.agglomeration());
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
        Dcoeff[0] = 1.;
        double factor = 1. / (2.0 * numVertices);
        VectorOfAveragedDiffusionCoefficients[agglomerate].axpy(factor,Dcoeff);
      }
    }
    // obtain quadrature order
    const int quadOrder = uLocal.order() + wLocal.order();

    { // element integral
      QuadratureType quadrature(entity, quadOrder);
      const size_t numQuadraturePoints = quadrature.nop();
      for (size_t pt = 0; pt < numQuadraturePoints; ++pt) {
        //! [Compute local contribution of operator]
        const typename QuadratureType::CoordinateType &x =
          quadrature.point(pt);
        //
        const double weight = quadrature.weight(pt)
          * geometry.integrationElement(x);

        DomainRangeType vu;
        uLocal.evaluate(quadrature[pt], vu);
        DomainJacobianRangeType du;
        uLocal.jacobian(quadrature[pt], du);

        // compute mass contribution (studying linear case so linearizing around zero)
        RangeRangeType avu(0);
        model().source(quadrature[pt], vu, du, avu);
        avu *= weight;
        // add to local functional wLocal.axpy( quadrature[ pt ], avu );

        RangeJacobianRangeType adu(0);
        // apply diffusive flux
        model().diffusiveFlux(quadrature[pt], vu, du, adu);
        adu *= weight;
        // add to local function
        wLocal.axpy(quadrature[pt], avu, adu);
        //! [Compute local contribution of operator]
      }
    }
    if (model().hasNeumanBoundary()) {
      if (!entity.hasBoundaryIntersections())
        continue;

      const IntersectionIteratorType iitend = dfSpace.gridPart().iend(
          entity);
      for (IntersectionIteratorType iit = dfSpace.gridPart().ibegin(
            entity); iit != iitend; ++iit) // looping over intersections
      {
        const IntersectionType &intersection = *iit;
        if (!intersection.boundary())
          continue;
        Dune::FieldVector<int, RangeRangeType::dimension> components(1);
        bool hasDirichletComponent = model().isDirichletIntersection(
            intersection, components);
        const typename IntersectionType::Geometry &intersectionGeometry =
          intersection.geometry();
        FaceQuadratureType quadInside(dfSpace.gridPart(), intersection,
            quadOrder, FaceQuadratureType::INSIDE);
        const size_t numQuadraturePoints = quadInside.nop();
        for (size_t pt = 0; pt < numQuadraturePoints; ++pt) {
          const typename FaceQuadratureType::LocalCoordinateType &x =
            quadInside.localPoint(pt);
          double weight = quadInside.weight(pt)
            * intersectionGeometry.integrationElement(x);
          DomainRangeType vu;
          uLocal.evaluate(quadInside[pt], vu);
          RangeRangeType alpha(0);
          model().alpha(quadInside[pt], vu, alpha);
          alpha *= weight;
          for (int k = 0; k < RangeRangeType::dimension; ++k)
            if (hasDirichletComponent && components[k])
              alpha[k] = 0;
          wLocal.axpy(quadInside[pt], alpha);
        }
      }
    }
  } // element loop end
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
          wLocal[r] += VectorOfAveragedDiffusionCoefficients[agglomerate][0]  * stabMatrix[r][c] * uLocal[c];
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
  std::cout << "starting assembly\n";
  Dune::Timer timer;
  typedef typename JacobianOperator::LocalMatrixType LocalMatrixType;
  typedef typename DomainDiscreteFunctionSpaceType::BasisFunctionSetType DomainBasisFunctionSetType;
  typedef typename RangeDiscreteFunctionSpaceType::BasisFunctionSetType RangeBasisFunctionSetType;

  const DomainDiscreteFunctionSpaceType &domainSpace = jOp.domainSpace();
  const RangeDiscreteFunctionSpaceType &rangeSpace = jOp.rangeSpace();

  // std::cout << "   in assembly: matrix stencil    " << timer.elapsed() << std::endl;;
  Dune::Fem::DiagonalStencil<DomainDiscreteFunctionSpaceType,
    RangeDiscreteFunctionSpaceType> stencil(domainSpace, rangeSpace);
  // std::cout << "   in assembly: matrix reserve   " << timer.elapsed() << std::endl;;
  jOp.reserve(stencil);
  // std::cout << "   in assembly: matrix clear    " << timer.elapsed() << std::endl;;
  jOp.clear();

  // std::cout << "   in assembly: setting up vectors    " << timer.elapsed() << std::endl;;
  const int domainBlockSize = domainSpace.localBlockSize; // is equal to 1 for scalar functions
  // Note the following is much! too large since it assumes e.g. all vertices in one polygon
  std::size_t maxSize = domainSpace.blockMapper().maxNumDofs() * domainBlockSize;
  std::vector<typename DomainLocalFunctionType::RangeType> phi;
  phi.reserve(maxSize);
  std::vector<typename DomainLocalFunctionType::JacobianRangeType> dphi;
  dphi.reserve(maxSize);
  // const int rangeBlockSize = rangeSpace.localBlockSize; // is equal to 1 for scalar functions
  // std::vector<typename RangeLocalFunctionType::RangeType> rphi(
  //     rangeSpace.blockMapper().maxNumDofs() * rangeBlockSize);
  // std::vector<typename RangeLocalFunctionType::JacobianRangeType> rdphi(
  //     rangeSpace.blockMapper().maxNumDofs() * rangeBlockSize);

  std::vector<double> polygonareas(rangeSpace.agglomeration().size(), 0.0);
  std::vector<RangeRangeType> VectorOfAveragedDiffusionCoefficients (rangeSpace.agglomeration().size(), RangeRangeType(0));
  std::vector<RangeRangeType> VectorOfAveragedLinearlisedDiffusionCoefficients (rangeSpace.agglomeration().size(), RangeRangeType(0));

  RangeRangeType Dcoeff(0);
  RangeRangeType LinDcoeff(0);
  const GridPartType &gridPart = rangeSpace.gridPart();


  // std::cout << "   in assembly: computing index set   " << timer.elapsed() << std::endl;;
  // const Dune::Vem::AgglomerationIndexSet<GridPartType> agIndexSet(
  //     rangeSpace.agglomeration());
  const auto &agIndexSet    = rangeSpace.blockMapper().indexSet();
  const auto &agglomeration = rangeSpace.agglomeration();
  std::cout << "   in assembly: start element loop size=" << rangeSpace.gridPart().grid().size(0) << " time=  " << timer.elapsed() << std::endl;;
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
        //!!! assuming a scalar diffusion coefficient of the form
        //!!! a(x,u).grad u
        //!!! diffusiveFlux(x,v,dv,ret)
        //!!! returns D(x,v,dv) = a(x,u).grad u
        DomainJacobianRangeType grad1;
        grad1[0][0] = 1.;
        grad1[0][1] = 0.;
        RangeJacobianRangeType a(0);
        model().diffusiveFlux(LocalPoint, vu, grad1, a);
        Dcoeff[0] = a[0][0];
        //!!! hack for accessing derivative of diffusion coefficient:
        //!!! assuming a scalar diffusion coefficient
        //!!! linDiffusiveFlux(u,du,x,v,dv,ret)
        //!!! returns d/du D(u,du,x).dv = d/du a(x,u).grad v
        model().linDiffusiveFlux(vu, dvu, LocalPoint, vu, grad1, a);
        LinDcoeff[0] = a[0][0];
        //std::cout << "LinDcoeff= " << LinDcoeff[0] << std::endl;
        // Each vertex visited twice in looping around the edges, so we divide by 2
        double factor = 1./(2. * numVertices);
        VectorOfAveragedDiffusionCoefficients[agglomerate].axpy(factor,Dcoeff);
        VectorOfAveragedLinearlisedDiffusionCoefficients[agglomerate].axpy(factor,LinDcoeff);
      }
    }

    polygonareas[agglomerate] += geometry.volume();
    const DomainBasisFunctionSetType &domainBaseSet = jLocal.domainBasisFunctionSet();
    const RangeBasisFunctionSetType &rangeBaseSet = jLocal.rangeBasisFunctionSet();
    const unsigned int domainNumBasisFunctions = domainBaseSet.size();
    phi.resize(domainNumBasisFunctions);
    dphi.resize(domainNumBasisFunctions);

    QuadratureType quadrature(entity, domainSpace.order() + rangeSpace.order());
    const std::size_t numQuadraturePoints = quadrature.nop();
    for (std::size_t pt = 0; pt < numQuadraturePoints; ++pt)
    {
      //! [Assembling the local matrix]
      const typename QuadratureType::CoordinateType &x = quadrature.point(pt);
      const double weight = quadrature.weight(pt) * geometry.integrationElement(x);

      // evaluate all basis functions at given quadrature point
      // ... optimize for equal domain and range spaces
      domainBaseSet.evaluateAll(quadrature[pt], phi);
      //    rangeBaseSet.evaluateAll(quadrature[pt], rphi);

      // evaluate jacobians of all basis functions at given quadrature point
      domainBaseSet.jacobianAll(quadrature[pt], dphi);
      //    rangeBaseSet.jacobianAll(quadrature[pt], rdphi);

      // get value for linearization
      DomainRangeType u0;
      DomainJacobianRangeType jacU0;
      uLocal.evaluate(quadrature[pt], u0);
      uLocal.jacobian(quadrature[pt], jacU0);

      RangeRangeType aphi(0);
      RangeJacobianRangeType adphi(0);
      for (unsigned int localCol = 0; localCol < domainNumBasisFunctions; ++localCol) {
        // if mass terms or right hand side is present
        model().linSource(u0, jacU0, quadrature[pt], phi[localCol], dphi[localCol], aphi);

        // if gradient term is present
        model().linDiffusiveFlux(u0, jacU0, quadrature[pt], phi[localCol], dphi[localCol], adphi);

        // get column object and call axpy method
        // jLocal.column(localCol).axpy(rphi, rdphi, aphi, adphi, weight);
        jLocal.column(localCol).axpy(phi, dphi, aphi, adphi, weight);
      } // basis function loop in stiffness computation
      //! [Assembling the local matrix]
    } // stiffness quadrature loop

    if (model().hasNeumanBoundary() && entity.hasBoundaryIntersections())
    {
      for (const IntersectionType &intersection : intersections(
            static_cast<typename GridPartType::GridViewType>(gridPart),
            entity)) {
        if (!intersection.boundary())
          continue;

        Dune::FieldVector<int, RangeRangeType::dimension> components(1);
        bool hasDirichletComponent = model().isDirichletIntersection(
            intersection, components);

        const typename IntersectionType::Geometry &intersectionGeometry =
          intersection.geometry();
        FaceQuadratureType quadInside(gridPart, intersection,
            domainSpace.order() + rangeSpace.order(),
            FaceQuadratureType::INSIDE);
        const std::size_t numQuadraturePoints = quadInside.nop();
        for (size_t pt = 0; pt < numQuadraturePoints; ++pt) {
          const typename FaceQuadratureType::LocalCoordinateType &x =
            quadInside.localPoint(pt);
          double weight = quadInside.weight(pt)
            * intersectionGeometry.integrationElement(x);
          DomainRangeType u0;
          uLocal.evaluate(quadInside[pt], u0);
          domainBaseSet.evaluateAll(quadInside[pt], phi);
          for (unsigned int localCol = 0;
              localCol < domainNumBasisFunctions; ++localCol) {
            RangeRangeType alpha(0);
            model().linAlpha(u0, quadInside[pt],
                phi[localCol], alpha);
            for (int k = 0; k < RangeRangeType::dimension; ++k) {
              if (hasDirichletComponent && components[k])
                alpha[k] = 0;
            }
            jLocal.column(localCol).axpy(phi, alpha, weight);
          } // basis function loop
        } // quadrature loop over edge
      } // intersection loop
    } // Neumann boundary loop
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
        {
          auto a = VectorOfAveragedDiffusionCoefficients[agglomerate][0];
          auto b = stabMatrix[r][c];
          if (!a==a) std::cout << "a=" << a << std::endl;
          if (!b==b) std::cout << "b=" << b << std::endl;
          jLocal.add(r, c, VectorOfAveragedDiffusionCoefficients[agglomerate][0]  * stabMatrix[r][c]);
        }
      const int nE = agIndexSet.numPolyVertices(entity, GridPartType::dimension);
      const DomainLocalFunctionType uLocal = u.localFunction(entity);
      for (std::size_t c = 0; c < stabMatrix.cols(); ++c)
        for (std::size_t r = 0; r < stabMatrix.rows(); ++r)
          for (std::size_t ccc = 0; ccc < stabMatrix.cols(); ++ccc)
          {
            auto a = VectorOfAveragedLinearlisedDiffusionCoefficients [agglomerate][0];
            auto b = stabMatrix[r][ccc];
            auto c = uLocal[ccc] / (nE);
            if (!a==a) std::cout << "a=" << a << std::endl;
            if (!b==b) std::cout << "b=" << b << std::endl;
            if (!c==c) std::cout << "c=" << c << std::endl;
            jLocal.add(r, c,VectorOfAveragedLinearlisedDiffusionCoefficients [agglomerate][0]  * stabMatrix[r][ccc] * uLocal[ccc] / (nE));
          }
      stabilization[agglomerate] = true;
    }
  }
  jOp.communicate();
  std::cout << "   in assembly: final    " << timer.elapsed() << std::endl;;
}
#endif // #ifndef VEMELLIPTIC_HH
