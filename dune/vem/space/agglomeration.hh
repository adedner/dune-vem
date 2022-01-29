#ifndef DUNE_VEM_SPACE_AGGLOMERATION_HH
#define DUNE_VEM_SPACE_AGGLOMERATION_HH

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
#include <dune/vem/space/interpolation.hh>
#include <dune/vem/space/interpolate.hh>

#include <dune/vem/space/test.hh>

#include <dune/vem/misc/vector.hh>

namespace Dune {

    namespace Vem {

        // Internal Forward Declarations
        // -----------------------------

        template<class FunctionSpace, class GridPart>
        class AgglomerationVEMSpace;



        // IsAgglomerationVEMSpace
        // -----------------------

        template<class DiscreteFunctionSpace>
        struct IsAgglomerationVEMSpace
                : std::integral_constant<bool, false> {
        };

        template<class FunctionSpace, class GridPart>
        struct IsAgglomerationVEMSpace<AgglomerationVEMSpace<FunctionSpace, GridPart> >
                : std::integral_constant<bool, true> {
        };



        // AgglomerationVEMSpaceTraits
        // ---------------------------

        template<class FunctionSpace, class GridPart>
        struct AgglomerationVEMSpaceTraits {
            friend class AgglomerationVEMSpace<FunctionSpace, GridPart>;

            typedef AgglomerationVEMSpace<FunctionSpace, GridPart> DiscreteFunctionSpaceType;

            typedef FunctionSpace FunctionSpaceType;
            typedef GridPart GridPartType;

            static const int codimension = 0;

        private:
            typedef typename GridPartType::template Codim<codimension>::EntityType EntityType;

        public:
            typedef Dune::Fem::FunctionSpace<
                    typename FunctionSpace::DomainFieldType, typename FunctionSpace::RangeFieldType,
                    GridPartType::dimension, 1
            > ScalarShapeFunctionSpaceType;
            typedef Dune::Fem::OrthonormalShapeFunctionSet< ScalarShapeFunctionSpaceType > ScalarShapeFunctionSetType;
            typedef Fem::VectorialShapeFunctionSet< Fem::ShapeFunctionSetProxy< ScalarShapeFunctionSetType >, typename ScalarShapeFunctionSpaceType::RangeType> ShapeFunctionSetType;
            typedef BoundingBoxBasisFunctionSet< GridPartType, ShapeFunctionSetType > ScalarBBBasisFunctionSetType;
            typedef VEMBasisFunctionSet <EntityType, ScalarBBBasisFunctionSetType> ScalarBasisFunctionSetType;
            typedef Fem::VectorialBasisFunctionSet<ScalarBasisFunctionSetType, typename FunctionSpaceType::RangeType> BasisFunctionSetType;

            typedef Hybrid::IndexRange<int, FunctionSpaceType::dimRange> LocalBlockIndices;
            typedef VemAgglomerationIndexSet <GridPartType> AgglomerationIndexSetType;
            typedef AgglomerationDofMapper <GridPartType, AgglomerationIndexSetType> BlockMapperType;

            template<class DiscreteFunction, class Operation = Fem::DFCommunicationOperation::Copy>
            struct CommDataHandle {
                typedef Operation OperationType;
                typedef Fem::DefaultCommunicationHandler <DiscreteFunction, Operation> Type;
            };
        };



        // AgglomerationVEMSpace
        // ---------------------

        template<class FunctionSpace, class GridPart>
        class AgglomerationVEMSpace
        : public Fem::DiscreteFunctionSpaceDefault<AgglomerationVEMSpaceTraits<FunctionSpace, GridPart> >
        {
            typedef AgglomerationVEMSpace<FunctionSpace, GridPart> ThisType;
            typedef Fem::DiscreteFunctionSpaceDefault <AgglomerationVEMSpaceTraits<FunctionSpace, GridPart>> BaseType;

        public:
            typedef typename BaseType::Traits Traits;

            typedef Agglomeration <GridPart> AgglomerationType;

            typedef typename Traits::AgglomerationIndexSetType AgglomerationIndexSetType;
            typedef AgglomerationVEMInterpolation <AgglomerationIndexSetType> AgglomerationInterpolationType;
            typedef typename Traits::ScalarShapeFunctionSetType ScalarShapeFunctionSetType;

        public:
            typedef typename BaseType::BasisFunctionSetType BasisFunctionSetType;

            typedef typename BaseType::BlockMapperType BlockMapperType;

            typedef typename BaseType::EntityType EntityType;
            typedef typename BaseType::GridPartType GridPartType;
            typedef Dune::Fem::FunctionSpace<double, double, GridPartType::dimensionworld - 1, 1> EdgeFSType;
            typedef Dune::Fem::OrthonormalShapeFunctionSet<EdgeFSType> EdgeShapeFunctionSetType;
            typedef typename BasisFunctionSetType::DomainFieldType DomainFieldType;
            typedef typename BasisFunctionSetType::DomainType DomainType;
            typedef typename GridPart::template Codim<0>::EntityType ElementType;
            typedef typename GridPart::template Codim<0>::EntitySeedType ElementSeedType;

#if 1 // FemQuads
            typedef Dune::Fem::ElementQuadrature<GridPartType, 0> Quadrature0Type;
            typedef Dune::Fem::ElementQuadrature<GridPartType, 1> Quadrature1Type;
#else
            typedef Dune::Fem::ElementQuadrature<GridPartType,0,Dune::Fem::HighOrderQuadratureTraits> Quadrature0Type;
            typedef Dune::Fem::ElementQuadrature<GridPartType,1,Dune::Fem::HighOrderQuadratureTraits> Quadrature1Type;
#endif

            typedef typename Traits::ScalarBasisFunctionSetType::HessianMatrixType HessianMatrixType;

            typedef DynamicMatrix<typename BasisFunctionSetType::DomainFieldType> Stabilization;

            using BaseType::gridPart;

            enum { hasLocalInterpolate = false };
            // static const int polynomialOrder = polOrder_;

            // for interpolation
            struct InterpolationType {
                InterpolationType(const AgglomerationIndexSetType &indexSet, const EntityType &element) noexcept
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
            //!TS: change to vector of vectors
            AgglomerationVEMSpace(AgglomerationType &agglomeration,
                const unsigned int polOrder,
                const typename AgglomerationIndexSetType::TestSpacesType &testSpaces,
                int basisChoice,
                bool edgeInterpolation)
            : BaseType(agglomeration.gridPart()),
              edgeInterpolation_(edgeInterpolation),
              agIndexSet_(agglomeration, testSpaces),
              blockMapper_(agIndexSet_, agIndexSet_.dofsPerCodim()),
              interpolation_(blockMapper().indexSet(), polOrder, basisChoice != 3),
              scalarShapeFunctionSet_(Dune::GeometryType(Dune::GeometryType::cube, GridPart::dimension),polOrder),
              edgeShapeFunctionSet_(Dune::GeometryType(Dune::GeometryType::cube, GridPart::dimension - 1),
                   agIndexSet_.maxEdgeDegree()),
              basisChoice_(basisChoice),
              polOrder_(polOrder),
              valueProjections_(new Vector<
                  typename Traits::ScalarBasisFunctionSetType::ValueProjection>()),
              jacobianProjections_(new Vector<
                  typename Traits::ScalarBasisFunctionSetType::JacobianProjection>()),
              hessianProjections_(new Vector<
                  typename Traits::ScalarBasisFunctionSetType::HessianProjection>()),
              stabilizations_(new Vector<Stabilization>()),
              useOnb_(basisChoice == 2),
              counter_(0),
              useThreads_(Fem::MPIManager::numThreads())
            {
              std::cout << "using " << useThreads_ << " threads\n";
              if (basisChoice != 3)
                this->agglomeration().onbBasis(order());
              update(true);
            }
            AgglomerationVEMSpace(const AgglomerationVEMSpace&) = delete;
            AgglomerationVEMSpace& operator=(const AgglomerationVEMSpace&) = delete;
            ~AgglomerationVEMSpace() {}
            void update(bool first=false)
            {
              // std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
              ++counter_;
              // std::cout << "space counter: " << counter_ << "agglo counter: " << agglomeration().counter() << std::endl;
              if (agglomeration().counter()<counter_)
              {
                // std::cout << "updating base classes\n";
                agIndexSet_.update();
              }

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

            const BasisFunctionSetType basisFunctionSet(const EntityType &entity) const
            {
              const std::size_t agglomerate = agglomeration().index(entity);
              assert(agglomerate<valueProjections().size());
              assert(agglomerate<jacobianProjections().size());
              assert(agglomerate<hessianProjections().size());
              // scalar ONB Basis proxy
              typename Traits::ShapeFunctionSetType scalarShapeFunctionSet(&scalarShapeFunctionSet_);
              // scalar BB Basis
              typename Traits::ScalarBBBasisFunctionSetType bbScalarBasisFunctionSet(entity, agglomerate,
                agglomeration().boundingBoxes(),
                useOnb_, std::move(scalarShapeFunctionSet));
              // vectorial extended VEM Basis
              typename Traits::ScalarBasisFunctionSetType scalarBFS(entity, agglomerate,
                             valueProjections_, jacobianProjections_, hessianProjections_,
                             std::move(bbScalarBasisFunctionSet));
              return BasisFunctionSetType(std::move(scalarBFS));
            }
            const typename Traits::ScalarBasisFunctionSetType scalarBasisFunctionSet(const EntityType &entity) const
            {
              const std::size_t agglomerate = agglomeration().index(entity);
              assert(agglomerate<valueProjections().size());
              assert(agglomerate<jacobianProjections().size());
              assert(agglomerate<hessianProjections().size());
              // scalar ONB Basis proxy
              typename Traits::ShapeFunctionSetType scalarShapeFunctionSet(&scalarShapeFunctionSet_);
              // scalar BB Basis
              typename Traits::ScalarBBBasisFunctionSetType bbScalarBasisFunctionSet(entity, agglomerate,
                agglomeration().boundingBoxes(),
                useOnb_, std::move(scalarShapeFunctionSet));
              // vectorial extended VEM Basis
              return typename Traits::ScalarBasisFunctionSetType(entity, agglomerate,
                              valueProjections_, jacobianProjections_, hessianProjections_,
                              std::move(bbScalarBasisFunctionSet));
            }

            BlockMapperType &blockMapper() const { return blockMapper_; }

            // extra interface methods

            static constexpr bool continuous() noexcept { return false; }

            static constexpr bool continuous(const typename BaseType::IntersectionType &) noexcept { return false; }

            int order(const EntityType &) const { return polOrder_; }
            int order() const { return polOrder_; }

            static constexpr Fem::DFSpaceIdentifier type() noexcept { return Fem::GenericSpace_id; }

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

            template <class T>
            using Vector = Std::vector<T>;
            auto& valueProjections() const { return *valueProjections_; }
            auto& jacobianProjections() const { return *jacobianProjections_; }
            auto& hessianProjections() const { return *hessianProjections_; }
            auto& stabilizations() const { return *stabilizations_; }
            std::shared_ptr<Vector<typename Traits::ScalarBasisFunctionSetType::ValueProjection>> valueProjections_;
            std::shared_ptr<Vector<typename Traits::ScalarBasisFunctionSetType::JacobianProjection>> jacobianProjections_;
            std::shared_ptr<Vector<typename Traits::ScalarBasisFunctionSetType::HessianProjection>> hessianProjections_;
            std::shared_ptr<Vector<Stabilization>> stabilizations_;

            void buildProjections(const Std::vector<Std::vector<ElementSeedType> > &entitySeeds,
                                  unsigned int start, unsigned int end);

            // issue with making these const: use of delete default constructor in some python bindings...
            bool edgeInterpolation_;
            AgglomerationIndexSetType agIndexSet_;
            mutable BlockMapperType blockMapper_;
            AgglomerationInterpolationType interpolation_;
            ScalarShapeFunctionSetType scalarShapeFunctionSet_;
            EdgeShapeFunctionSetType edgeShapeFunctionSet_;
            unsigned int polOrder_;
            int basisChoice_;
            const bool useOnb_;
            std::size_t counter_;
            int useThreads_;
        };


        // Implementation of AgglomerationVEMSpace
        // ---------------------------------------

        template<class FunctionSpace, class GridPart>
        inline void AgglomerationVEMSpace<FunctionSpace, GridPart>::buildProjections(
              const Std::vector<Std::vector<ElementSeedType> > &entitySeeds,
              unsigned int start, unsigned int end )
        {
          int polOrder = order();
          Std::vector<int> orders = agIndexSet_.orders();
          const std::size_t numShapeFunctions = scalarShapeFunctionSet_.size();
          const std::size_t numHessShapeFunctions =
                polOrder==1? 1:
                std::min(numShapeFunctions,
                   Dune::Fem::OrthonormalShapeFunctions<DomainType::dimension>::
                   size(std::max(orders[2], polOrder - 2))
           );
          std::size_t numGradShapeFunctions =
                   std::min(numShapeFunctions,
                   Dune::Fem::OrthonormalShapeFunctions<DomainType::dimension>::
                   size(std::max(orders[1], polOrder - 1))
           );
          const std::size_t numInnerShapeFunctions = orders[0] < 0 ? 0 :
                  Dune::Fem::OrthonormalShapeFunctions<DomainType::dimension>::
                  size(orders[0]);

          const std::size_t numGradConstraints = numGradShapeFunctions;
          const std::size_t edgeTangentialSize = Dune::Fem::OrthonormalShapeFunctions<1>::
               size( agIndexSet_.edgeOrders()[0] + 1);
          std::size_t edgeNormalSize = agIndexSet_.template order2size<1>(1);

          // set up matrices used for constructing gradient, value, and edge projections
          // Note: the code is set up with the assumption that the dofs suffice to compute the edge projection
          DynamicMatrix <DomainFieldType> D, C, constraintValueProj, constraintGradProj, leastSquaresGradProj, RHSleastSquaresGrad, Hp, HpGrad, HpHess, HpInv, HpGradInv, HpHessInv;
          DynamicMatrix <DomainType> R;
          DynamicMatrix<typename Traits::ScalarBasisFunctionSetType::HessianMatrixType> P;

          LeftPseudoInverse <DomainFieldType> pseudoInverse(numShapeFunctions);

          AgglomerationInterpolationType interpolation(blockMapper().indexSet(), polOrder, basisChoice_ != 3);

          // start iteration over all polygons
          for (std::size_t agglomerate = start; agglomerate < end; ++agglomerate)
          {
            const auto &bbox = blockMapper_.indexSet().boundingBox(agglomerate);
            const std::size_t numDofs = blockMapper().numDofs(agglomerate);

            const int numEdges = agIndexSet_.subAgglomerates(agglomerate, AgglomerationIndexSetType::dimension - 1);

            D.resize(numDofs, numShapeFunctions, 0);
            C.resize(numShapeFunctions, numDofs, 0);
            Hp.resize(numShapeFunctions, numShapeFunctions, 0);
            HpGrad.resize(numGradShapeFunctions, numGradShapeFunctions, 0);
            HpHess.resize(numHessShapeFunctions, numHessShapeFunctions, 0);
            R.resize(numGradConstraints, numDofs, DomainType(0));
            P.resize(numHessShapeFunctions, numDofs, 0);
            constraintValueProj.resize(numInnerShapeFunctions, numShapeFunctions, 0);
            constraintGradProj.resize(numGradConstraints,numGradShapeFunctions,0);
            leastSquaresGradProj.resize( numEdges*(edgeNormalSize + edgeTangentialSize ) , 2*numGradShapeFunctions,0);
            RHSleastSquaresGrad.resize( numDofs, leastSquaresGradProj.rows(), 0);

            // iterate over the triangles of this polygon
            for (const ElementSeedType &entitySeed : entitySeeds[agglomerate])
            {
              const ElementType &element = gridPart().entity(entitySeed);
              const auto geometry = element.geometry();
              Quadrature0Type quadrature(element, 3 * polOrder);

              // get the bounding box monomials and apply all dofs to them
              BoundingBoxBasisFunctionSet <GridPart, ScalarShapeFunctionSetType> shapeFunctionSet(element, agglomerate,
                      agglomeration().boundingBoxes(),
                      useOnb_, scalarShapeFunctionSet_);
              interpolation(shapeFunctionSet, D);

              // compute mass matrices Hp, HpGrad, and the gradient matrices G^l
              for (std::size_t qp = 0; qp < quadrature.nop(); ++qp) {
                const DomainFieldType weight =
                        geometry.integrationElement(quadrature.point(qp)) * quadrature.weight(qp);
                shapeFunctionSet.evaluateEach(quadrature[qp], [&](std::size_t alpha, FieldVector<DomainFieldType, 1> phi) {
                  shapeFunctionSet.evaluateEach(quadrature[qp], [&](std::size_t beta, FieldVector<DomainFieldType, 1> psi) {
                    Hp[alpha][beta] += phi[0] * psi[0] * weight;
                    if (alpha < numGradShapeFunctions && beta < numGradShapeFunctions) // basis set is hierarchic so we can compute HpGrad using the order p shapeFunctionSet
                    {
                      HpGrad[alpha][beta] += phi[0] * psi[0] * weight;
                    }
                    if (alpha < numGradConstraints && beta < numGradShapeFunctions )
                    {
                      constraintGradProj[alpha][beta] += phi[0] * psi[0] * weight;
                    }
                    if (alpha < numHessShapeFunctions && beta < numHessShapeFunctions)
                      HpHess[alpha][beta] += phi[0] * psi[0] * weight;
                    if (alpha < numInnerShapeFunctions)
                      constraintValueProj[alpha][beta] += phi[0] * psi[0] * weight;
                  });
                });
              } // quadrature loop
            } // loop over triangles in agglomerate

            // finished agglomerating all auxiliary matrices
            // now compute projection matrices and stabilization for the given polygon

            // volume of polygon
            DomainFieldType H0 = blockMapper_.indexSet().volume(agglomerate);

            auto &valueProjection = valueProjections()[agglomerate];
            auto &jacobianProjection = jacobianProjections()[agglomerate];
            auto &hessianProjection = hessianProjections()[agglomerate];
            valueProjection.resize(numShapeFunctions);
            jacobianProjection.resize(numShapeFunctions);
            hessianProjection.resize(numShapeFunctions);
            for (std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha) {
              valueProjection[alpha].resize(numDofs, 0);
              jacobianProjection[alpha].resize(numDofs, DomainType(0));
              hessianProjection[alpha].resize(numDofs, HessianMatrixType(0));
            }

            // type def for standard vector (to pick up re size for Hessian projection)
            // need to resize Hessian projection

            // re implementation of the value projection
            auto leastSquaresMinimizer = LeastSquares(D, constraintValueProj);
            Std::vector<DomainFieldType> b(numDofs, 0), d(numInnerShapeFunctions, 0);

            for ( std::size_t beta = 0; beta < numDofs; ++beta )
            {
              auto colValueProjection = vectorizeMatrixCol( valueProjection, beta );
              // set up vectors b and d needed for least squares
              b[ beta ] = 1;
              if( beta >= numDofs - numInnerShapeFunctions )
                d[ beta - numDofs + numInnerShapeFunctions] = H0;

              colValueProjection = leastSquaresMinimizer.solve(b, d);

              if (beta >= numDofs - numInnerShapeFunctions)
                d[beta - numDofs + numInnerShapeFunctions] = 0;

              b[beta] = 0;
            }
            HpInv = Hp;
            HpInv.invert();
            HpGradInv = HpGrad;
            HpGradInv.invert();
            HpHessInv = HpHess;
            HpHessInv.invert();

            //////////////////////////////////////////////////////////////////////////
            /// GradientProjecjtion //////////////////////////////////////////////////
            //////////////////////////////////////////////////////////////////////////

            Std::vector< Std::vector<unsigned int>> fullMask;

            // iterate over the triangles of this polygon
            int counter = 0;
            int counter2 = 0;

            for (const ElementSeedType &entitySeed : entitySeeds[agglomerate])
            {
              const ElementType &element = gridPart().entity(entitySeed);
              const auto geometry = element.geometry();
              const auto &refElement = ReferenceElements<typename GridPart::ctype, GridPart::dimension>::general( element.type());
              Std::vector<Dune::DynamicMatrix<double> > edgePhiVector(2);
              // get the bounding box monomials and apply all dofs to them
              BoundingBoxBasisFunctionSet <GridPart, ScalarShapeFunctionSetType>
                  shapeFunctionSet(element, agglomerate,
                  agglomeration().boundingBoxes(),
                  useOnb_, scalarShapeFunctionSet_);

              auto vemBasisFunction = scalarBasisFunctionSet(element);

              // compute the boundary terms for the gradient projection
              for (const auto &intersection : intersections(gridPart(), element))
              {
                // ignore edges inside the given polygon
                if (!intersection.boundary() && (agglomeration().index(intersection.outside()) == agglomerate))
                  continue;
                assert(intersection.conforming());
                auto normal = intersection.centerUnitOuterNormal();
                typename Traits::ScalarBasisFunctionSetType::HessianMatrixType factorTN, factorNN;
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
                auto globalNormal = normal;
                if (intersection.neighbor()) // we need to check the orientation of the normal
                  if (blockMapper_.indexSet().index(intersection.inside()) >
                      blockMapper_.indexSet().index(intersection.outside()))
                    globalNormal *= -1;

                Std::vector<Std::vector<unsigned int>>
                  mask(2,Std::vector<unsigned int>(0)); // contains indices with Phi_mask[i] is attached to given edge
                edgePhiVector[0].resize(agIndexSet_.edgeSize(0), agIndexSet_.edgeSize(0), 0);
                edgePhiVector[1].resize(agIndexSet_.edgeSize(1), agIndexSet_.edgeSize(1), 0);
                interpolation(intersection, edgeShapeFunctionSet_, edgePhiVector, mask);
                //!!! std::cout << "edgePhiVector[0].size:" << edgePhiVector[0].size() << std::endl;
                if (edgePhiVector[0].size() > 0)
                {
                  edgePhiVector[0].invert();
                  //!!! std::cout << "mask[1].size:" << mask[1].size() << " " << "agIndexSet_.edgeSize(1):" << agIndexSet_.edgeSize(1) << std::endl;
                  if (mask[1].size() > agIndexSet_.edgeSize(1))
                  { // need to take tangential derivatives at vertices into account
                    assert(mask[0].size() == agIndexSet_.edgeSize(0)+2);
                    auto A = edgePhiVector[0];
                    edgePhiVector[0].resize(agIndexSet_.edgeSize(0), mask[0].size(), 0);
                    // vertex basis functions (values)
                    for (std::size_t j=0;j<agIndexSet_.edgeSize(0);++j)
                    {
                      edgePhiVector[0][j][0] = A[j][0];
                      edgePhiVector[0][j][3] = A[j][2];
                    }
                    // vertex basis functions (tangential derivatives)
                    for (std::size_t j=0;j<agIndexSet_.edgeSize(0);++j)
                    {
                      edgePhiVector[0][j][1] = A[j][1]*tau[0];
                      edgePhiVector[0][j][2] = A[j][1]*tau[1];
                      edgePhiVector[0][j][4] = A[j][3]*tau[0];
                      edgePhiVector[0][j][5] = A[j][3]*tau[1];
                    }
                    for (std::size_t i=6;i<mask[0].size();++i)
                      for (std::size_t j=0;j<agIndexSet_.edgeSize(0);++j)
                      {
                        assert( i-2 < A[j].size() );
                        edgePhiVector[0][j][i] = A[j][i-2];
                      }
                  }
                }
                if (edgePhiVector[1].size() > 0)
                {
                  edgePhiVector[1].invert();
                  if (mask[1].size() > agIndexSet_.edgeSize(1))
                  {
                    assert(mask[1].size() == agIndexSet_.edgeSize(1)+2);
                    auto A = edgePhiVector[1];
                    edgePhiVector[1].resize(agIndexSet_.edgeSize(1), mask[1].size(), 0);
                    std::size_t i=0;
                    // vertex basis functions
                    for (;i<4;i+=2)
                    {
                      for (std::size_t j=0;j<agIndexSet_.edgeSize(1);++j)
                      {
                        edgePhiVector[1][j][i]   = A[j][i/2]*globalNormal[0];
                        edgePhiVector[1][j][i+1] = A[j][i/2]*globalNormal[1];
                      }
                    }
                    for (;i<mask[1].size();++i)
                      for (std::size_t j=0;j<agIndexSet_.edgeSize(1);++j)
                        edgePhiVector[1][j][i] = A[j][i-2];
                  }
                }
                /* WARNING WARNING WARNING
                 * This is a horrible HACK and needs to be revised:
                 * It might be necessary to flip the vertex entries in the masks around
                 * due to a twist in the intersection.
                 * At the moment this is done by checking that the
                 * interpolation properties for phiEdge is satisfied - if not
                 * the mask is switch and the test is repeated!
                 * WARNING WARNING WARNING */
                {
                  Std::vector<double> lambda(numDofs);
                  bool succ = true;
                  std::fill(lambda.begin(), lambda.end(), 0);
                  PhiEdge <GridPartType, Dune::DynamicMatrix<double>, EdgeShapeFunctionSetType>
                          phiEdge(gridPart(), intersection, edgePhiVector[0], edgeShapeFunctionSet_,
                                  0); // behaves like Phi_mask[i] restricted to edge
                  interpolation(element, phiEdge, lambda);
                  // test with phi_mask[0] (vertex) has the correct interpolation result:
                  succ = std::abs(lambda[mask[0][0]] - 1) < 1e-10;
                  DomainType otherTau = element.geometry().corner(
                               refElement.subEntity(intersection.indexInInside(),1,1,2)
                             );
                  otherTau -= element.geometry().corner(
                                refElement.subEntity(intersection.indexInInside(),1,0,2)
                              );
                  otherTau /= otherTau.two_norm();
                  if (agIndexSet_.vertexOrders()[0]>=0) // vertices might have to be flipped
                  {
                    assert( (succ && otherTau*tau>0) || (!succ && otherTau*tau<0) );
                    if (otherTau*tau<0)
                    {
                      if (mask[1].size() > agIndexSet_.edgeSize(1))
                      {
                        std::swap(mask[0][0], mask[0][3]); // the HACK
                        std::swap(mask[0][1], mask[0][4]); // the HACK
                        std::swap(mask[0][2], mask[0][5]); // the HACK
                        std::swap(mask[1][0], mask[1][2]); // the HACK
                        std::swap(mask[1][1], mask[1][3]); // the HACK
                      }
                      else
                        std::swap(mask[0][0], mask[0][1]); // the HACK
                    }
                  }
                  std::fill(lambda.begin(), lambda.end(), 0);
                  interpolation(element, phiEdge, lambda);
                  succ = std::abs(lambda[mask[0][0]] - 1) < 1e-10;
                  assert(succ);
                }

                // now compute int_e Phi_mask[i] m_alpha
                Quadrature1Type quadrature(gridPart(), intersection, 3 * polOrder, Quadrature1Type::INSIDE);
                for (std::size_t qp = 0; qp < quadrature.nop(); ++qp)
                {
                  auto x = quadrature.localPoint(qp);
                  auto y = intersection.geometryInInside().global(x);
                  const DomainFieldType weight = intersection.geometry().integrationElement(x) * quadrature.weight(qp);
                  shapeFunctionSet.evaluateEach(y, [&](std::size_t alpha, FieldVector<DomainFieldType, 1> phi) {
                  /*
                      std::cout << alpha << "," << qp << "," << x << ":"
                                << " " << Dune::Fem::OrthonormalShapeFunctions<DomainType::dimension>::size( agIndexSet_.edgeOrders()[0] )
                                << " " << edgePhiVector[0].size()
                                << " " << polOrder+1
                                << " " << edgeInterpolation_
                                << std::endl;
                   */
                      if (alpha < numGradShapeFunctions)
                      {
                        // evaluate each here for edge shape fns
                        // first check if we should be using interpolation (for the
                        // existing edge moments - or for H4 space)
                        auto normal = intersection.unitOuterNormal();
                        if (alpha < Dune::Fem::OrthonormalShapeFunctions<DomainType::dimension>::
                                    size( agIndexSet_.edgeOrders()[0] ) // have enough edge momentsa
                            || edgePhiVector[0].size() == polOrder+1    // interpolation is exact
                            || edgeInterpolation_)                      // user want interpolation no matter what
                        {
                          edgeShapeFunctionSet_.evaluateEach(x, [&](std::size_t beta, FieldVector<DomainFieldType, 1> psi) {
                            if (beta < edgePhiVector[0].size())
                              for (std::size_t s = 0; s < mask[0].size(); ++s)// note that edgePhi is the transposed of the basis transform matrix
                                R[alpha][mask[0][s]].axpy(edgePhiVector[0][beta][s] * psi[0] * phi[0] * weight, normal);
                          });
                        }
                        else // use value projection
                        {
                          auto factor = normal;
                          factor *= phi[0]*weight;
                          vemBasisFunction.axpy(y, factor, R[alpha]);
                        }
                      }
                      if (alpha < numHessShapeFunctions && agIndexSet_.edgeSize(1) > 0)
                      {
#if 1 // can be replaced by integration by parts version further down
                        auto jit = intersection.geometry().jacobianInverseTransposed(x);
                        //jacobian each here for edge shape fns
                        edgeShapeFunctionSet_.jacobianEach(x, [&](std::size_t beta,
                                                                  FieldMatrix<DomainFieldType, 1, 1> dpsi) {
                            if (beta < edgePhiVector[0].size()) {
                              // note: the edgeShapeFunctionSet is defined over
                              // the reference element of the edge so the jit has
                              // to be applied here
                              Dune::FieldVector<double, 2> gradPsi;
                              jit.mv(dpsi[0], gradPsi);
                              double gradPsiDottau = gradPsi * tau;
                              assert(std::abs(gradPsiDottau - dpsi[0][0] / h) < 1e-8);
                              for (std::size_t s = 0; s < mask[0].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                              {
                                P[alpha][mask[0][s]].axpy(edgePhiVector[0][beta][s] * gradPsiDottau * phi[0] * weight,
                                                          factorTN);
                              }
                            }
                        });
#endif
                      } // alpha < numHessSF
                      if (alpha < numHessShapeFunctions && agIndexSet_.edgeSize(1) > 0) {
                        edgeShapeFunctionSet_.evaluateEach(x, [&](std::size_t beta, FieldVector<DomainFieldType, 1> psi) {
                          if (beta < edgePhiVector[1].size())
                            for (std::size_t s = 0; s < mask[1].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                              P[alpha][mask[1][s]].axpy(edgePhiVector[1][beta][s] * psi[0] * phi[0] * weight, factorNN);
                        });
                      } // alpha < numHessSF and can compute normal derivative
                  });
                  ////////////////////////////////
                  #if 0
                  shapeFunctionSet.evaluateEach(y, [&](std::size_t alpha, FieldVector<DomainFieldType, 1> phi) {
                      if (alpha < numGradShapeFunctions)
                      {
                        edgeShapeFunctionSet_.evaluateEach(x, [&](std::size_t beta, FieldVector<DomainFieldType, 1> psi) {
                          if (beta < edgeNormalSize)
                          {
                            // assert( edgeNormalSize==0 || edgeNormalSize == edgePhiVector[1].size() );
                            // assert( edgeNormalSize == agIndexSet_.edgeSize(1) );
                            leastSquaresGradProj[counter+beta][alpha]                          += psi[0] * phi[0] * weight * globalNormal[0];
                            leastSquaresGradProj[counter+beta][alpha + numGradShapeFunctions ] += psi[0] * phi[0] * weight * globalNormal[1];
                          }
                          if (beta < edgeTangentialSize)
                          { // initialize counter2 = numEdges+edgeNormalSize
                            leastSquaresGradProj[numEdges*edgeNormalSize + counter2+beta][alpha]                          += psi[0] * phi[0] * weight * tau[0];
                            leastSquaresGradProj[numEdges*edgeNormalSize + counter2+beta][alpha + numGradShapeFunctions ] += psi[0] * phi[0] * weight * tau[1];
                          }
                        });
                     }
                  });
                  #endif
                  auto jit = intersection.geometry().jacobianInverseTransposed(x);
                  edgeShapeFunctionSet_.evaluateEach(x, [&](std::size_t alpha, FieldVector<DomainFieldType, 1> phi) {
                    if (alpha < edgeTangentialSize) // note: in contrast to the previous loop alpha is now the test space
                    {
                      edgeShapeFunctionSet_.jacobianEach(x, [&](std::size_t beta, FieldMatrix<DomainFieldType, 1, 1> dpsi) {
                        if (beta < edgePhiVector[0].size())
                        {
                          Dune::FieldVector<double, 2> gradPsi;
                          jit.mv(dpsi[0], gradPsi);
                          double gradPsiDottau = gradPsi * tau;
                          // ??? this fails on moving grids for some reason
                          // (gdb) print gradPsiDottau    $1 = 23.509848240825679
                          // (gdb) print dpsi[0][0] / h   $2 = 23.581436077925769
                          // assert(std::abs(gradPsiDottau - dpsi[0][0] / h) < 1e-8);
                          for (std::size_t s = 0; s < mask[0].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                          {
                            RHSleastSquaresGrad[ mask[0][s] ][ numEdges*edgeNormalSize + counter2+alpha ] += edgePhiVector[0][beta][s] * gradPsiDottau * phi[0] * weight;
                          }
                        }
                      });
                    }
                   });
#if 0 // implement tangential derivative using integration by part on edge - also need point evaluations below to be turned on
                  shapeFunctionSet.jacobianEach( y, [ & ] ( std::size_t alpha, FieldMatrix< DomainFieldType, 1, 2 > dphi ) {
                     if (alpha<numHessShapeFunctions)
                     {
                        double dphids = dphi[0]*tau;
                        edgeShapeFunctionSet_.evaluateEach( x, [ & ] ( std::size_t beta, FieldVector< DomainFieldType, 1 > psi ) {
                            //!!! TODO test if beta not too large
                            for (int s=0;s<mask[0].size();++s) // note that edgePhi is the transposed of the basis transform matrix
                              P[alpha][mask[0][s]].axpy( -edgePhiVector[0][beta][s]*psi[0]*dphids*weight, factorTN);
                        } );
                     }
                  } );
#endif
                } // quadrature loop
#if 0 // implement tangential derivative using integration by part on edge - also need inner integral above to be turned on
                auto x = Dune::FieldVector<double,1>{1.};
                auto y = intersection.geometryInInside().global(x);
                shapeFunctionSet.evaluateEach( y, [ & ] ( std::size_t alpha, FieldVector< DomainFieldType, 1 > phi ) {
                   if (alpha<numHessShapeFunctions)
                   {
                      edgeShapeFunctionSet_.evaluateEach( x, [ & ] ( std::size_t beta, FieldVector< DomainFieldType, 1 > psi ) {
                         //!!! TODO test if beta not too large
                         for (int s=0;s<mask[0].size();++s) // note that edgePhi is the transposed of the basis transform matrix
                           P[alpha][mask[0][s]].axpy( edgePhiVector[0][beta][s]*psi[0]*phi[0], factorTN);
                     } );
                   }
                 } );
                x = Dune::FieldVector<double,1>{0.};
                y = intersection.geometryInInside().global(x);
                shapeFunctionSet.evaluateEach( y, [ & ] ( std::size_t alpha, FieldVector< DomainFieldType, 1 > phi ) {
                   if (alpha<numHessShapeFunctions)
                   {
                      edgeShapeFunctionSet_.evaluateEach( x, [ & ] ( std::size_t beta, FieldVector< DomainFieldType, 1 > psi ) {
                         //!!! TODO test if beta not too large
                         for (int s=0;s<mask[0].size();++s) // note that edgePhi is the transposed of the basis transform matrix
                           P[alpha][mask[0][s]].axpy( -edgePhiVector[0][beta][s]*psi[0]*phi[0], factorTN);
                     } );
                   }
                 } );
#endif
                // store the masks for each edge
                fullMask.push_back(mask[1]);
                counter  += agIndexSet_.edgeSize(1);
                counter2 += edgeTangentialSize;
              } // loop over intersections


              Quadrature0Type quadrature(element, 3 * polOrder);
              for (std::size_t qp = 0; qp < quadrature.nop(); ++qp)
              {
                const DomainFieldType weight =
                        geometry.integrationElement(quadrature.point(qp)) * quadrature.weight(qp);
                shapeFunctionSet.jacobianEach(quadrature[qp],
                                 [&](std::size_t alpha, typename ScalarShapeFunctionSetType::JacobianRangeType gradPhi) {
                    // Note: the shapeFunctionSet is defined in physical space so
                    // the jit is not needed here
                    // R[alpha][j]  -=  Pi phi_j  grad(m_alpha) * weight
                    if (alpha < numGradConstraints) {
                      gradPhi[0] *= -weight;
                      vemBasisFunction.axpy(quadrature[qp], gradPhi[0], R[alpha]);
                    }
                });
              } // quadrature loop

            } // loop over triangles in agglomerate


            // least squares
            if ( leastSquaresGradProj.size() == 0 )
            {
              assert( numGradConstraints == numGradShapeFunctions );
              // need to compute value projection first
              // now compute projection by multiplying with inverse mass matrix
              for (std::size_t alpha = 0; alpha < numGradShapeFunctions; ++alpha)
                for (std::size_t i = 0; i < numDofs; ++i)
                {
                  jacobianProjection[alpha][i] = 0;
                  for (std::size_t beta = 0; beta < numGradShapeFunctions; ++beta)
                    jacobianProjection[alpha][i].axpy(HpGradInv[alpha][beta], R[beta][i]);
                }
            }
            else {
              BlockMatrix constraintBlockMatrix = blockMatrix(constraintGradProj, 2);
              auto leastSquaresMinimizerGradient = leastSquares(leastSquaresGradProj, constraintBlockMatrix);
              // auto leastSquaresMinimizerGradient = leastSquares(leastSquaresGradProj);

              // std::cout << "ls grad proj:" << leastSquaresGradProj.rows() << "," << leastSquaresGradProj.cols() << " constraint:" << constraintBlockMatrix.rows() << "," << constraintBlockMatrix.cols() << std::endl;
              for (std::size_t beta = 0; beta < numDofs; ++beta)
              {
                VectorizeMatrixCol d = vectorizeMatrixCol(R, beta);
                VectorizeMatrixCol colGradProjection = vectorizeMatrixCol(jacobianProjection, beta);

#if 0 // memory access issues with C^1-conf space - but not needed anymore anyway
                // TODO: set RHS for the normal derivatives in loop using mask[1] then remove fullMask
                std::size_t counter = 0;
                bool finished = false;
                for (std::size_t e = 0; e < fullMask.size(); ++e)
                {
                  for (std::size_t i = 0; i < fullMask[e].size(); ++i, ++counter)
                  {
                    if (fullMask[e][i] == beta) {
                      RHSleastSquaresGrad[ beta ][ counter ] = 1;
                      finished = true;
                      break;
                    }
                  }
                  if (finished) break;
                }
#endif
                colGradProjection = leastSquaresMinimizerGradient.solve( RHSleastSquaresGrad[ beta ] , d );
              }
            }

            /////////////////////////////////////////////////////////////////////
            // HessianProjection ////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////

            // iterate over the triangles of this polygon (for Hessian projection)
            for (const ElementSeedType &entitySeed : entitySeeds[agglomerate])
            {
              const ElementType &element = gridPart().entity(entitySeed);
              const auto geometry = element.geometry();
              const auto &refElement = ReferenceElements<typename GridPart::ctype, GridPart::dimension>::general(element.type());

              // get the bounding box monomials and apply all dofs to them
              BoundingBoxBasisFunctionSet <GridPart, ScalarShapeFunctionSetType>
                shapeFunctionSet(element, agglomerate, agglomeration().boundingBoxes(),
                                 useOnb_, scalarShapeFunctionSet_);
              auto vemBasisFunction = scalarBasisFunctionSet(element);

              // compute the boundary terms for the gradient projection
              if (agIndexSet_.edgeSize(1) == 0)
              {
                for (const auto &intersection : intersections(gridPart(), element))
                {
                  // ignore edges inside the given polygon
                  if (!intersection.boundary() && (agglomeration().index(intersection.outside()) == agglomerate))
                    continue;
                  assert(intersection.conforming());
                  auto normal = intersection.centerUnitOuterNormal();

                  // change to compute boundary term in Hessian Projection
                  // now compute int_e Phi_mask[i] m_alpha
                  Quadrature1Type quadrature(gridPart(), intersection, 3 * polOrder, Quadrature1Type::INSIDE);
                  for (std::size_t qp = 0; qp < quadrature.nop(); ++qp)
                  {
                    auto x = quadrature.localPoint(qp);
                    auto y = intersection.geometryInInside().global(x);
                    const DomainFieldType weight = intersection.geometry().integrationElement(x) * quadrature.weight(qp);
                    shapeFunctionSet.evaluateEach(y, [&](std::size_t alpha, FieldVector<DomainFieldType, 1> phi) {
                        if (alpha < numHessShapeFunctions)
                        {
                          DomainType factor = normal;
                          factor *= weight * phi[0];
                          // Version 1: full gradient
                          // vemBasisFunction.axpy( quadrature[qp], factor, P[alpha] );
                          // Version 2: add normal part
                          vemBasisFunction.axpy(y, normal, factor, P[alpha]);
                          // Version 2a: add tangential part
                          // vemBasisFunction.axpy( y, DomainType{normal[1],-normal[0]}, factor, P[alpha] );
                        }
                    });
                  } // quadrature loop
                } // loop over intersections
              }
              Quadrature0Type quadrature(element, 3 * polOrder);
              for (std::size_t qp = 0; qp < quadrature.nop(); ++qp)
              {
                const DomainFieldType weight = geometry.integrationElement(quadrature.point(qp)) * quadrature.weight(qp);
                shapeFunctionSet.jacobianEach(quadrature[qp],
                              [&](std::size_t alpha, typename ScalarShapeFunctionSetType::JacobianRangeType gradPhi) {
                    // Note: the shapeFunctionSet is defined in physical space so
                    // the jit is not needed here
                    // P[alpha][j] -= Pi grad phi_j grad(m_alpha) * weight
                    if (alpha < numHessShapeFunctions)
                    {
                      // P[alpha] vector of hessians i.e. use axpy with type DynamicVector <HessianMatrixType>
                      gradPhi *= -weight;
                      vemBasisFunction.axpy(quadrature[qp], gradPhi[0], P[alpha]);
                    }
                });

              } // quadrature loop
            } // loop over triangles in agglomerate

            // need to compute value projection first
            // now compute projection by multiplying with inverse mass matrix
            for (std::size_t alpha = 0; alpha < numHessShapeFunctions; ++alpha)
              for (std::size_t i = 0; i < numDofs; ++i)
              {
                hessianProjection[alpha][i] = 0;
                for (std::size_t beta = 0; beta < numHessShapeFunctions; ++beta)
                  hessianProjection[alpha][i].axpy(HpHessInv[alpha][beta], P[beta][i]);
              }

            /////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////
            // stabilization matrix
            Stabilization S(numDofs, numDofs, 0);
            for (std::size_t i = 0; i < numDofs; ++i)
              S[i][i] = DomainFieldType(1);
            for (std::size_t i = 0; i < numDofs; ++i)
              for (std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha)
                for (std::size_t j = 0; j < numDofs; ++j) {
                  S[i][j] -= D[i][alpha] * valueProjection[alpha][j];
                  // std::cout << "space:" << i << " " << alpha << " " << j << "   "
                  //   << D[i][alpha] << " " << valueProjection[alpha][j]
                  //   << " -> " << S[i][j] << std::endl;
                }
            Stabilization &stabilization = stabilizations()[agglomerate];
            stabilization.resize(numDofs, numDofs, 0);
            for (std::size_t i = 0; i < numDofs; ++i)
              for (std::size_t j = 0; j < numDofs; ++j) {
                for (std::size_t k = 0; k < numDofs; ++k)
                  stabilization[i][j] += S[k][i] * S[k][j];
                // stabilization[ i ][ j ] += S[ k ][ i ] * std::max(1.,stabScaling[k]) * S[ k ][ j ];
                // std::cout << "   " << i << " " << j << " "
                //   << stabilization[i][j] << " " << S[i][j] << std::endl;
              }
          } // iterate over polygons

        } // build projections

    } // namespace Vem



    namespace Fem {

        namespace Capabilities {
            template<class FunctionSpace, class GridPart>
            struct hasInterpolation<Vem::AgglomerationVEMSpace<FunctionSpace, GridPart> > {
                static const bool v = false;
            };
        }

    } // namespace Fem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_AGGLOMERATION_HH
