#ifndef DUNE_VEM_SPACE_DEFAULT_HH
#define DUNE_VEM_SPACE_DEFAULT_HH

#include <cassert>
#include <utility>

#include <dune/fem/common/hybrid.hh>

#include <dune/fem/quadrature/elementquadrature.hh>
#include <dune/fem/space/common/commoperations.hh>
#include <dune/fem/space/common/defaultcommhandler.hh>
#include <dune/fem/space/common/discretefunctionspace.hh>
#include <dune/fem/space/common/functionspace.hh>
#include <dune/fem/space/basisfunctionset/vectorial.hh>
#include <dune/fem/space/shapefunctionset/orthonormal.hh>
#include <dune/fem/space/shapefunctionset/proxy.hh>
#include <dune/fem/space/shapefunctionset/vectorial.hh>
#include <dune/fem/space/common/capabilities.hh>

#include <dune/vem/space/indexset.hh>
#include <dune/vem/agglomeration/dofmapper.hh>
#include <dune/vem/misc/compatibility.hh>
#include <dune/vem/misc/pseudoinverse.hh>
#include <dune/vem/misc/leastSquares.hh>
#include <dune/vem/misc/matrixWrappers.hh>
#include <dune/vem/space/basisfunctionset.hh>
#include <dune/vem/space/interpolate.hh>

#include <dune/vem/misc/vector.hh>

namespace Dune
{

  namespace Vem
  {
    template<class VemTraits>
    class DefaultAgglomerationVEMSpace
    : public Fem::DiscreteFunctionSpaceDefault< VemTraits >
    {
      typedef DefaultAgglomerationVEMSpace< VemTraits > ThisType;
      typedef Fem::DiscreteFunctionSpaceDefault< VemTraits > BaseType;

    public:
      typedef typename BaseType::Traits Traits;
      typedef typename BaseType::GridPartType GridPartType;

      typedef Agglomeration<GridPartType> AgglomerationType;

      typedef typename Traits::IndexSetType IndexSetType;
      typedef typename Traits::template InterpolationType<Traits> AgglomerationInterpolationType;
      typedef typename Traits::BasisSetsType BasisSetsType;

    public:
      typedef typename BaseType::BasisFunctionSetType BasisFunctionSetType;
      typedef typename BaseType::BlockMapperType BlockMapperType;

      typedef typename BaseType::EntityType EntityType;
      typedef typename BasisSetsType::EdgeShapeFunctionSetType EdgeShapeFunctionSetType;
      typedef typename BasisFunctionSetType::DomainFieldType DomainFieldType;
      typedef typename BasisFunctionSetType::DomainType DomainType;

      typedef typename BasisFunctionSetType::RangeType RangeType;
      typedef typename BasisFunctionSetType::JacobianRangeType JacobianRangeType;
      typedef typename BasisFunctionSetType::HessianRangeType HessianRangeType;
      typedef typename GridPartType::template Codim<0>::EntityType ElementType;
      typedef typename GridPartType::template Codim<0>::EntitySeedType ElementSeedType;

      static const int dimDomain = Traits::dimDomain;

      typedef Dune::Fem::ElementQuadrature<GridPartType, 0> Quadrature0Type;
      typedef Dune::Fem::ElementQuadrature<GridPartType, 1> Quadrature1Type;

      typedef typename Traits::ScalarBasisFunctionSetType::HessianMatrixType HessianMatrixType;

      typedef DynamicMatrix<typename BasisFunctionSetType::DomainFieldType> Stabilization;

      using BaseType::gridPart;

      enum { hasLocalInterpolate = false };

      // for interpolation
      struct InterpolationType {
          InterpolationType(const IndexSetType &indexSet, const EntityType &element) noexcept
                  : inter_(indexSet), element_(element) {}
          template<class U, class V>
          void operator()(const U &u, V &v)
          { inter_(element_, u, v); }
          AgglomerationInterpolationType inter_;
          const EntityType &element_;
      };

      // basisChoice:
      // 1: use onb for inner moments but not for computing projections
      // 2: use onb for both the inner moments and computation of projection
      // 3: don't use onb at all
      DefaultAgglomerationVEMSpace(AgglomerationType &agglomeration,
          const unsigned int polOrder,
          const typename IndexSetType::TestSpacesType &testSpaces,
          int basisChoice,
          bool edgeInterpolation)
      : BaseType(agglomeration.gridPart()),
        polOrder_(polOrder),
        basisChoice_(basisChoice),
        useOnb_(basisChoice == 2),
        edgeInterpolation_(edgeInterpolation),
        agIndexSet_(agglomeration, testSpaces),
        blockMapper_(agIndexSet_, agIndexSet_.dofsPerCodim()),
        interpolation_(blockMapper().indexSet(), polOrder, basisChoice != 3),
        basisSets_(polOrder_, agIndexSet_.maxEdgeDegree(), useOnb_),
        counter_(0),
        useThreads_(Fem::MPIManager::numThreads()),
        valueProjections_(new Vector<
            typename Traits::ScalarBasisFunctionSetType::ValueProjection>()),
        jacobianProjections_(new Vector<
            typename Traits::ScalarBasisFunctionSetType::JacobianProjection>()),
        hessianProjections_(new Vector<
            typename Traits::ScalarBasisFunctionSetType::HessianProjection>()),
        stabilizations_(new Vector<Stabilization>())
      {
        std::cout << "using " << useThreads_ << " threads\n";
        if (basisChoice != 3)
          this->agglomeration().onbBasis(order());
        update(true);
      }
      DefaultAgglomerationVEMSpace(const DefaultAgglomerationVEMSpace&) = delete;
      DefaultAgglomerationVEMSpace& operator=(const DefaultAgglomerationVEMSpace&) = delete;
      ~DefaultAgglomerationVEMSpace() {}

      void update(bool first=false)
      {
        ++counter_;
        if (agglomeration().counter()<counter_)
          agIndexSet_.update();

        // these are the matrices we need to compute
        valueProjections().resize(agglomeration().size());
        jacobianProjections().resize(agglomeration().size());
        hessianProjections().resize(agglomeration().size());
        stabilizations().resize(agglomeration().size());

        Std::vector<Std::vector<ElementSeedType> > entitySeeds(agglomeration().size());
        for (const ElementType &element : elements(gridPart(), Partitions::interiorBorder))
          entitySeeds[agglomeration().index(element)].push_back(element.seed());

        if (first) // use single thread at start to guarantee quadratures build
        {
          buildProjections(entitySeeds,0,agglomeration().size());
        }
        else
        {
          const double threadSize = agglomeration().size() / useThreads_;
          std::vector<unsigned int> threads(useThreads_+1,0);
          threads[0] = 0;
          for (std::size_t t=1; t < useThreads_; ++t)
            threads[t] = int(t*threadSize);
          threads[useThreads_] = agglomeration().size();
          Fem::MPIManager :: run( [this, &entitySeeds, &threads]() {
                unsigned int start = threads[Fem::MPIManager::thread()];
                unsigned int end   = threads[Fem::MPIManager::thread()+1];
                buildProjections(entitySeeds,start,end);
              });
        }
      }

      const typename Traits::ScalarBasisFunctionSetType scalarBasisFunctionSet(const EntityType &entity) const
      {
        const std::size_t agglomerate = agglomeration().index(entity);
        assert(agglomerate<valueProjections().size());
        assert(agglomerate<jacobianProjections().size());
        assert(agglomerate<hessianProjections().size());
        return typename Traits::ScalarBasisFunctionSetType(entity, agglomerate,
                        valueProjections_, jacobianProjections_, hessianProjections_,
                        basisSets_.basisFunctionSet(agglomeration(), entity)
               );
      }
      const BasisFunctionSetType basisFunctionSet(const EntityType &entity) const
      {
        // if constexpr (vectorSpace) return scalarBasisFunctionSet(entity); else
        return BasisFunctionSetType( std::move(scalarBasisFunctionSet(entity)) );
      }

      BlockMapperType &blockMapper() const { return blockMapper_; }

      // extra interface methods
      static constexpr bool continuous() noexcept { return false; }
      static constexpr bool continuous(const typename BaseType::IntersectionType &) noexcept { return false; }
      static constexpr Fem::DFSpaceIdentifier type() noexcept { return Fem::GenericSpace_id; }

      int order(const EntityType &) const { return polOrder_; }
      int order() const { return polOrder_; }

      // implementation-defined methods
      const AgglomerationType &agglomeration() const { return agIndexSet_.agglomeration(); }
      AgglomerationType &agglomeration() { return agIndexSet_.agglomeration(); }

      const Stabilization &stabilization(const EntityType &entity) const
      {
        assert( agglomeration().index(entity)<stabilizations().size());
        return stabilizations()[agglomeration().index(entity)];
      }

      //////////////////////////////////////////////////////////
      // Non-interface methods (used in DirichletConstraints) //
      //////////////////////////////////////////////////////////
      /** \brief return local interpolation for given entity
       *
       *  \param[in]  entity  grid part entity
       */
      InterpolationType interpolation(const EntityType &entity) const
      {
        return InterpolationType(blockMapper().indexSet(), entity);
      }

      AgglomerationInterpolationType interpolation() const
      {
        return interpolation_;
      }

    private:
      template <int codim>
      static std::size_t sizeONB(std::size_t order)
      {
        return Dune::Fem::OrthonormalShapeFunctions<DomainType::dimension - codim>:: size(order) *
               Traits::ScalarBasisFunctionSetType::RangeType::dimension;
      }

      template <class T>
      using Vector = Std::vector<T>;
      auto& valueProjections() const { return *valueProjections_; }
      auto& jacobianProjections() const { return *jacobianProjections_; }
      auto& hessianProjections() const { return *hessianProjections_; }
      auto& stabilizations() const { return *stabilizations_; }

      void buildProjections(const Std::vector<Std::vector<ElementSeedType> > &entitySeeds,
                            unsigned int start, unsigned int end);

      // issue with making these const: use of delete default constructor in some python bindings...
      unsigned int polOrder_;
      int basisChoice_;
      const bool useOnb_;
      bool edgeInterpolation_;
      IndexSetType agIndexSet_;
      mutable BlockMapperType blockMapper_;
      AgglomerationInterpolationType interpolation_;
      BasisSetsType basisSets_;
      std::size_t counter_;
      int useThreads_;
      std::shared_ptr<Vector<typename Traits::ScalarBasisFunctionSetType::ValueProjection>> valueProjections_;
      std::shared_ptr<Vector<typename Traits::ScalarBasisFunctionSetType::JacobianProjection>> jacobianProjections_;
      std::shared_ptr<Vector<typename Traits::ScalarBasisFunctionSetType::HessianProjection>> hessianProjections_;
      std::shared_ptr<Vector<Stabilization>> stabilizations_;
    };

    // Computation of  projections for DefaultAgglomerationVEMSpace
    // ------------------------------------------------------------

    template<class Traits>
    inline void DefaultAgglomerationVEMSpace<Traits> :: buildProjections(
          const Std::vector<Std::vector<ElementSeedType> > &entitySeeds,
          unsigned int start, unsigned int end )
    {
      int polOrder = order();
      typedef typename BasisSetsType::EdgeShapeFunctionSetType EdgeTestSpace;
      typedef typename BasisSetsType::ShapeFunctionSetType::FunctionSpaceType FunctionSpaceType;
      typedef typename FunctionSpaceType::DomainType DomainType;
      typedef typename FunctionSpaceType::RangeType RangeType;
      typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;
      typedef typename FunctionSpaceType::HessianRangeType HessianRangeType;
      const std::size_t dimDomain = DomainType::dimension;
      const std::size_t dimRange = RangeType::dimension;

      AgglomerationInterpolationType interpolation(blockMapper().indexSet(), polOrder, basisChoice_ != 3);

      const std::size_t baseRangeDimension = RangeType::dimension;
      assert( RangeType::dimension == Traits::baseRangeDimension );

      Std::vector<int> orders = agIndexSet_.orders();
      const std::size_t numShapeFunctions = basisSets_.size(0);
      const std::size_t numGradShapeFunctions = basisSets_.size(1);
      const std::size_t numHessShapeFunctions = basisSets_.size(2);
      const std::size_t numInnerShapeFunctions = basisSets_.innerTestSize();

      // set up matrices used for constructing gradient, value, and edge projections
      // Note: the code is set up with the assumption that the dofs suffice to compute the edge projection
      //       Is this still the case?

      // Mass matrices and their inverse: Hp, HpGrad, HpHess, HpInv, HpGradInv, HpHessInv
      DynamicMatrix<DomainFieldType> Hp, HpGrad, HpHess, HpInv, HpGradInv, HpHessInv;
      Hp.resize(numShapeFunctions, numShapeFunctions, 0);
      HpGrad.resize(numGradShapeFunctions, numGradShapeFunctions, 0);
      HpHess.resize(numHessShapeFunctions, numHessShapeFunctions, 0);

      // interpolation of basis function set used for least squares part of value projection
      DynamicMatrix<DomainFieldType> D;
      // constraint matrix for value projection
      DynamicMatrix<DomainFieldType> constraintValueProj;
      constraintValueProj.resize(numInnerShapeFunctions, numShapeFunctions, 0);
      // right hand sides and solvers for CLS for value projection (b: ls, d: constraints)
      Dune::DynamicVector<DomainFieldType> b, d;
      d.resize(numInnerShapeFunctions, 0);

      // matrices for edge projections
      Std::vector<Dune::DynamicMatrix<double> > edgePhiVector(2);
      edgePhiVector[0].resize(interpolation.edgeSize(0), interpolation.edgeSize(0), 0);
      edgePhiVector[1].resize(interpolation.edgeSize(1), interpolation.edgeSize(1), 0);

      // matrix for rhs of gradient and hessian projections
      DynamicMatrix<DomainFieldType> R;
      DynamicMatrix<DomainFieldType> P;
      std::vector<RangeType> phi0Values;

      // start iteration over all polygons
      for (std::size_t agglomerate = start; agglomerate < end; ++agglomerate)
      {
        const std::size_t numDofs = blockMapper().numDofs(agglomerate) * baseRangeDimension;

        phi0Values.resize(numDofs);

        const int numEdges = agIndexSet_.subAgglomerates(agglomerate, IndexSetType::dimension - 1);

        //////////////////////////////////////////////////////////////////////////////
        /// resize matrices that depend on the local number of degrees of freedom  ///
        //////////////////////////////////////////////////////////////////////////////

        auto &valueProjection = valueProjections()[agglomerate];
        auto &jacobianProjection = jacobianProjections()[agglomerate];
        auto &hessianProjection = hessianProjections()[agglomerate];
        valueProjection.resize(numShapeFunctions);
        jacobianProjection.resize(numGradShapeFunctions);
        hessianProjection.resize(numHessShapeFunctions);
        for (std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha)
          valueProjection[alpha].resize(numDofs, DomainFieldType(0));
        for (std::size_t alpha = 0; alpha < numGradShapeFunctions; ++alpha)
          jacobianProjection[alpha].resize(numDofs, DomainFieldType(0));
        for (std::size_t alpha = 0; alpha < numHessShapeFunctions; ++alpha)
          hessianProjection[alpha].resize(numDofs, DomainFieldType(0));

        // value projection CLS
        constraintValueProj = 0;
        D.resize(numDofs, numShapeFunctions, 0);
        b.resize(numDofs, 0);
        std::fill(d.begin(),d.end(),0);

        // rhs structures for gradient/hessian projection
        R.resize(numGradShapeFunctions, numDofs, 0);
        P.resize(numHessShapeFunctions, numDofs, 0);

        //////////////////////////////////////////////////////////////////////////
        /// compute L(B) and the mass matrices ///////////////////////////////////
        //////////////////////////////////////////////////////////////////////////

        Hp = 0;
        HpGrad = 0;
        HpHess = 0;
        for (const ElementSeedType &entitySeed : entitySeeds[agglomerate])
        {
          const ElementType &element = gridPart().entity(entitySeed);
          const auto geometry = element.geometry();
          Quadrature0Type quadrature(element, 2 * polOrder);

          // get the bounding box monomials and apply all dofs to them
          // GENERAL: these are the same as used as test function in 'interpolation'
          const auto &shapeFunctionSet = basisSets_.basisFunctionSet(agglomeration(), element);

          interpolation.interpolateBasis(element, shapeFunctionSet, D);

          // compute mass matrices Hp, HpGrad, and the gradient matrices G^l
          // TO DO change here to call shapeFunctionSet.jacobianEach and hessianEach from methods in hk
          for (std::size_t qp = 0; qp < quadrature.nop(); ++qp)
          {
            const DomainFieldType weight =
                    geometry.integrationElement(quadrature.point(qp)) * quadrature.weight(qp);
            shapeFunctionSet.evaluateEach(quadrature[qp], [&](std::size_t alpha, RangeType phi) {
              shapeFunctionSet.evaluateEach(quadrature[qp], [&](std::size_t beta, RangeType psi) {
                Hp[alpha][beta] += phi * psi * weight;
                if (alpha < numInnerShapeFunctions) // ??? numInnerShapeShapeFunctions from space implementation
                  constraintValueProj[alpha][beta] += phi * psi * weight;
              });
            });
            shapeFunctionSet.jacobianEach(quadrature[qp], [&](std::size_t alpha, JacobianRangeType phi) {
              shapeFunctionSet.jacobianEach(quadrature[qp], [&](std::size_t beta, JacobianRangeType psi) {
                  for (std::size_t i=0;i<dimRange;++i)
                    for (std::size_t j=0;j<dimDomain;++j)
                      HpGrad[alpha][beta] += phi[i][j] * psi[i][j] * weight;
              });
            });
            shapeFunctionSet.hessianEach(quadrature[qp], [&](std::size_t alpha, HessianRangeType phi) {
              shapeFunctionSet.hessianEach(quadrature[qp], [&](std::size_t beta, HessianRangeType psi) {
                  for (std::size_t i=0;i<dimRange;++i)
                    for (std::size_t j=0;j<dimDomain;++j)
                      for (std::size_t k=0;k<dimDomain;++k)
                        HpHess[alpha][beta] += phi[i][j][k] * psi[i][j][k] * weight;
              });
            });
          } // quadrature loop
        } // loop over triangles in agglomerate

        // compute inverse mass matrix
        // GENERAL: should we simply assume that the mass matrix is diagonal and not compute anything here?
        HpInv = Hp;
        HpInv.invert();
        HpGradInv = HpGrad;
        HpGradInv.invert();
        HpHessInv = HpHess;
        HpHessInv.invert();

        //////////////////////////////////////////////////////////////////////////
        /// ValueProjection /////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////

        DomainFieldType H0 = blockMapper_.indexSet().volume(agglomerate);
        auto leastSquaresMinimizer = LeastSquares(D, constraintValueProj);
        for ( std::size_t beta = 0; beta < numDofs; ++beta )
        {
          // set up vectors b (rhs for least squares)
          b[ beta ] = 1;

          // set up vector d (rhs for constraints)
          interpolation.valueL2constraints(beta, H0, D, d);
          if( beta >= numDofs - numInnerShapeFunctions )
            assert( std::abs( d[ beta - numDofs + numInnerShapeFunctions ] - H0 ) < 1e-13);

          // compite CLS solution and store in right column of 'valueProjection'
          auto colValueProjection = vectorizeMatrixCol( valueProjection, beta );
          colValueProjection = leastSquaresMinimizer.solve(b, d);

          b[beta] = 0;
        }

        //////////////////////////////////////////////////////////////////////////
        /// GradientProjection //////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////

        for (const ElementSeedType &entitySeed : entitySeeds[agglomerate])
        {
          const ElementType &element = gridPart().entity(entitySeed);
          const auto geometry = element.geometry();
          const auto &refElement = ReferenceElements<typename GridPartType::ctype, GridPartType::dimension>::general( element.type());

          // get the bounding box monomials and apply all dofs to them
          const auto &shapeFunctionSet = basisSets_.basisFunctionSet(agglomeration(), element);
          const typename BasisSetsType::EdgeShapeFunctionSetType &edgeShapeFunctionSet
                  = basisSets_.edgeBasisFunctionSet(agglomeration(), element);

          auto vemBasisFunction = scalarBasisFunctionSet(element);

          // compute the boundary terms for the gradient projection
          for (const auto &intersection : intersections(gridPart(), element))
          {
            // ignore edges inside the given polygon
            if (!intersection.boundary() && (agglomeration().index(intersection.outside()) == agglomerate))
              continue;
            assert(intersection.conforming());

            Std::vector<Std::vector<unsigned int>>
              mask(2,Std::vector<unsigned int>(0)); // contains indices with Phi_mask[i] is attached to given edge
            edgePhiVector[0] = 0;
            edgePhiVector[1] = 0;
            interpolation(intersection, edgeShapeFunctionSet, edgePhiVector, mask);

            auto normal = intersection.centerUnitOuterNormal();
            typename Dune::FieldMatrix<DomainFieldType,dimDomain,dimDomain> factorTN, factorNN;
            DomainType tau = intersection.geometry().corner(1);
            tau -= intersection.geometry().corner(0);
            double h = tau.two_norm();
            tau /= h;
            for (std::size_t i = 0; i < factorTN.rows; ++i)
              for (std::size_t j = 0; j < factorTN.cols; ++j)
              {
                factorTN[i][j] = 0.5 * (normal[i] * tau[j] + normal[j] * tau[i]);
                factorNN[i][j] = 0.5 * (normal[i] * normal[j] + normal[j] * normal[i]);
              }

            // now compute int_e Phi_mask[i] m_alpha
            Quadrature1Type quadrature(gridPart(), intersection, 2 * polOrder, Quadrature1Type::INSIDE);
            for (std::size_t qp = 0; qp < quadrature.nop(); ++qp)
            {
              auto x = quadrature.localPoint(qp);
              auto y = intersection.geometryInInside().global(x);
              const DomainFieldType weight = intersection.geometry().integrationElement(x) * quadrature.weight(qp);
              shapeFunctionSet.jacobianEach(y, [&](std::size_t alpha, JacobianRangeType phi) {
                  // evaluate each here for edge shape fns
                  // first check if we should be using interpolation (for the
                  // existing edge moments - or for H4 space)
                  // !!!!!
                  if (alpha < dimDomain*sizeONB<0>(agIndexSet_.edgeOrders()[0])       // have enough edge momentsa
                      || edgePhiVector[0].size() == dimRange*(polOrder+1)               // interpolation is exact
                      || edgeInterpolation_)                                 // user want interpolation no matter what
                  {
                    edgeShapeFunctionSet.evaluateEach(x, [&](std::size_t beta,
                          typename BasisSetsType::EdgeShapeFunctionSetType::RangeType psi)
                    {
                      if (beta < edgePhiVector[0].size())
                        for (std::size_t s=0; s<mask[0].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                          for (std::size_t i=0;i<dimRange;++i)
                            for (std::size_t j=0;j<dimDomain;++j)
                              R[alpha][mask[0][s]] += weight *
                                edgePhiVector[0][beta][s] * psi[i] * phi[i][j] * normal[j];
                    });
                  }
                  else // use value projection
                  {
                    assert(0);
                    vemBasisFunction.evaluateAll(y, phi0Values);
                    for (std::size_t s=0;s<numDofs;++s)
                      for (std::size_t i=0;i<dimRange;++i)
                        for (std::size_t j=0;j<dimDomain;++j)
                          R[alpha][s] += weight * phi0Values[s][i] * phi[i][j] * normal[j];
                  }
              });
#if 0
              hessShapeFunctionSet.evaluateEach(y, [&](std::size_t alpha, HessianRangeType phi) {
                  // compute the phi.tau boundary terms for the hessian projection using d/ds Pi^e
                  if ( interpolation.edgeSize(1) > 0 )
                  {
                    auto jit = intersection.geometry().jacobianInverseTransposed(x);
                    // jacobian each here for edge shape fns
                    edgeShapeFunctionSet.jacobianEach(x, [&](std::size_t beta,
                                                              typename EdgeTestSpace::JacobianRangeType dpsi) {
                        if (beta < edgePhiVector[0].size()) {
                          // note: the edgeShapeFunctionSet is defined over
                          // the reference element of the edge so the jit has
                          // to be applied here
                          Dune::FieldVector<double,1> gradHatPsiPhi;
                          dpsi.mtv(phi, gradHatPsiPhi);
                          DomainType gradPsiPhi;
                          jit.mv(gradHatPsiPhi, gradPsiPhi);
                          double gradPsiPhiDottau = gradPsiPhi * tau;
                          // assert(std::abs(gradPsiDottau - dpsi[r][0] / h) < 1e-8);
                          // GENERAL: this assumed that the Pi_0 part of Pi^e is not needed?
                          for (std::size_t s = 0; s < mask[0].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                            P[alpha][mask[0][s]].axpy(edgePhiVector[0][beta][s] * gradPsiPhiDottau * weight, factorTN);
                        }
                    });
                  } // alpha < numHessSF

                  // compute the phi.n boundary terms for the hessian projection in
                  // the case that there are dofs for the normal gradient on the edge
                  if ( interpolation.edgeSize(1) > 0 ) {
                    edgeShapeFunctionSet.evaluateEach(x, [&](std::size_t beta, typename EdgeTestSpace::RangeType psi) {
                      if (beta < edgePhiVector[1].size())
                        // GENERL: could use Pi_0 here as suggested in varying coeff paper
                        //         avoid having to use the gradient projection later for the hessian projection
                        for (std::size_t s = 0; s < mask[1].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                          P[alpha][mask[1][s]].axpy(edgePhiVector[1][beta][s] * psi*phi * weight, factorNN);
                    });
                  } // alpha < numHessSF and can compute normal derivative
              });
#endif
            } // quadrature loop
            // store the masks for each edge
          } // loop over intersections

          // Compute element part for the gradient projection
          Quadrature0Type quadrature(element, 2 * polOrder);
          for (std::size_t qp = 0; qp < quadrature.nop(); ++qp)
          {
            const DomainFieldType weight =
                    geometry.integrationElement(quadrature.point(qp)) * quadrature.weight(qp);
            vemBasisFunction.evaluateAll(quadrature[qp], phi0Values);
            shapeFunctionSet.divJacobianEach(quadrature[qp], [&](std::size_t alpha, RangeType divGradPhi) {
                // divGradPhi = RangeType = tr( D GradSF )
                for (std::size_t s=0; s<numDofs; ++s)
                  R[alpha][s] -= weight * phi0Values[s] * divGradPhi;
            });
          } // quadrature loop
        } // loop over triangles in agglomerate

        // now compute gradient projection by multiplying with inverse mass matrix
        for (std::size_t alpha = 0; alpha < numGradShapeFunctions; ++alpha)
          for (std::size_t i = 0; i < numDofs; ++i)
          {
            jacobianProjection[alpha][i] = 0;
            for (std::size_t beta = 0; beta < numGradShapeFunctions; ++beta)
              jacobianProjection[alpha][i] += HpGradInv[alpha][beta] * R[beta][i];
          }

        /////////////////////////////////////////////////////////////////////
        // HessianProjection ////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////
#if 0 // ???
        // iterate over the triangles of this polygon (for Hessian projection)
        for (const ElementSeedType &entitySeed : entitySeeds[agglomerate])
        {
          const ElementType &element = gridPart().entity(entitySeed);
          const auto geometry = element.geometry();

          // get the bounding box monomials and apply all dofs to them
          auto shapeFunctionSet = basisSets_.basisFunctionSet(agglomeration(), element);
          auto vemBasisFunction = scalarBasisFunctionSet(element);

          // compute the phi.n boundary terms for the hessian projection in
          // the case that there are no dofs for the normal gradient on the edge
          if (interpolation.edgeSize(1) == 0)
          {
            // GENERAL: more efficient to avoid this by using
            //          parital_n Pi^0 and compute that in the above intersection loop?
            //          Was this a bad idea?
            for (const auto &intersection : intersections(gridPart(), element))
            {
              // ignore edges inside the given polygon
              if (!intersection.boundary() && (agglomeration().index(intersection.outside()) == agglomerate))
                continue;
              assert(intersection.conforming());
              auto normal = intersection.centerUnitOuterNormal();

              // change to compute boundary term in Hessian Projection
              // now compute int_e Phi_mask[i] m_alpha
              Quadrature1Type quadrature(gridPart(), intersection, 2 * polOrder, Quadrature1Type::INSIDE);
              for (std::size_t qp = 0; qp < quadrature.nop(); ++qp)
              {
                auto x = quadrature.localPoint(qp);
                auto y = intersection.geometryInInside().global(x);
                const DomainFieldType weight = intersection.geometry().integrationElement(x) * quadrature.weight(qp);
                hessShapeFunctionSet.evaluateEach(y, [&](std::size_t alpha, auto phi) {
                    phi *= weight;
                    vemBasisFunction.axpy(y, phi, normal, P[alpha]);
                });
              } // quadrature loop
            } // loop over intersections
          }

          // Compute element part for the hessian projection
          // GENERAL: could use the value projection here by using additional integration by parts
          //          i.e., Pi^0 D^2 m
          Quadrature0Type quadrature(element, 2 * polOrder);
          for (std::size_t qp = 0; qp < quadrature.nop(); ++qp)
          {
            const DomainFieldType weight = geometry.integrationElement(quadrature.point(qp)) * quadrature.weight(qp);
            hessShapeFunctionSet.jacobianEach(quadrature[qp],
                          [&](std::size_t alpha, auto gradPhi) {
                // Note: the shapeFunctionSet is defined in physical space so
                // the jit is not needed here
                // P[alpha][j] -= Pi grad phi_j grad(m_alpha) * weight
                // P[alpha] vector of hessians i.e. use axpy with type DynamicVector <HessianMatrixType>
                gradPhi *= -weight;
                vemBasisFunction.axpy(quadrature[qp], gradPhi, P[alpha]);
            });

          } // quadrature loop
        } // loop over triangles in agglomerate

        // now compute hessian projection by multiplying with inverse mass matrix
        for (std::size_t alpha = 0; alpha < numHessShapeFunctions; ++alpha)
          for (std::size_t i = 0; i < numDofs; ++i)
          {
            hessianProjection[alpha][i] = 0;
            for (std::size_t beta = 0; beta < numHessShapeFunctions; ++beta)
              hessianProjection[alpha][i].axpy(HpHessInv[alpha][beta], P[beta][i]);
          }
#endif
        /////////////////////////////////////////////////////////////////////
        // stabilization matrix /////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////

        Stabilization S(numDofs, numDofs, 0);
        for (std::size_t i = 0; i < numDofs; ++i)
          S[i][i] = DomainFieldType(1);
        for (std::size_t i = 0; i < numDofs; ++i)
          for (std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha)
            for (std::size_t j = 0; j < numDofs; ++j)
              S[i][j] -= D[i][alpha] * valueProjection[alpha][j];
        Stabilization &stabilization = stabilizations()[agglomerate];
        stabilization.resize(numDofs, numDofs, 0);
        for (std::size_t i = 0; i < numDofs; ++i)
          for (std::size_t j = 0; j < numDofs; ++j)
          {
            for (std::size_t k = 0; k < numDofs; ++k)
              stabilization[i][j] += S[k][i] * S[k][j];
          }

      } // end iteration over polygonsa

    } // end build projections

  } // namespace Vem
} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_AGGLOMERATION_HH
