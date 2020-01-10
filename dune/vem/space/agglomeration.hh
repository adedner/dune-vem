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
#include <dune/vem/agglomeration/dgspace.hh>
#include <dune/vem/space/basisfunctionset.hh>
#include <dune/vem/space/interpolation.hh>
#include <dune/vem/space/interpolate.hh>

#include <dune/vem/space/test.hh>
// #include <dune/vem/misc/highorderquadratures.hh>

namespace Dune {

    namespace Vem {

        // Internal Forward Declarations
        // -----------------------------

        template<class FunctionSpace, class GridPart, int polOrder>
        class AgglomerationVEMSpace;



        // IsAgglomerationVEMSpace
        // -----------------------

        template<class DiscreteFunctionSpace>
        struct IsAgglomerationVEMSpace
                : std::integral_constant<bool, false> {
        };

        template<class FunctionSpace, class GridPart, int order>
        struct IsAgglomerationVEMSpace<AgglomerationVEMSpace<FunctionSpace, GridPart, order> >
                : std::integral_constant<bool, true> {
        };



        // AgglomerationVEMSpaceTraits
        // ---------------------------

        template<class FunctionSpace, class GridPart, int polOrder>
        struct AgglomerationVEMSpaceTraits {
            friend class AgglomerationVEMSpace<FunctionSpace, GridPart, polOrder>;

            typedef AgglomerationVEMSpace<FunctionSpace, GridPart, polOrder> DiscreteFunctionSpaceType;

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
            typedef AgglomerationDGSpaceTraits <ScalarShapeFunctionSpaceType, GridPart, polOrder> DGTraitsType;
            typedef typename DGTraitsType::ScalarShapeFunctionSetType ScalarShapeFunctionSetType;
            typedef typename DGTraitsType::BasisFunctionSetType ScalarBBBasisFunctionSetType;
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

        template<class FunctionSpace, class GridPart, int polOrder>
        class AgglomerationVEMSpace
                : public Fem::DiscreteFunctionSpaceDefault<AgglomerationVEMSpaceTraits<FunctionSpace, GridPart, polOrder> > {
            typedef AgglomerationVEMSpace<FunctionSpace, GridPart, polOrder> ThisType;
            typedef Fem::DiscreteFunctionSpaceDefault <AgglomerationVEMSpaceTraits<FunctionSpace, GridPart, polOrder>> BaseType;

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
            // typedef Dune::Fem::LegendreShapeFunctionSet<EdgeFSType,true> EdgeShapeFunctionSetType;
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


            typedef DynamicMatrix<typename BasisFunctionSetType::DomainFieldType> Stabilization;

            using BaseType::gridPart;

            enum { hasLocalInterpolate = false };
            static const int polynomialOrder = polOrder;

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
                const typename AgglomerationIndexSetType::TestSpacesType &testSpaces,
                int basisChoice)
                : BaseType(agglomeration.gridPart()),
                agIndexSet_(agglomeration, testSpaces),
                blockMapper_(agIndexSet_, agIndexSet_.dofsPerCodim()),
                interpolation_(blockMapper().indexSet(), polOrder, basisChoice != 3),
                scalarShapeFunctionSet_(Dune::GeometryType(Dune::GeometryType::cube, GridPart::dimension)),
                edgeShapeFunctionSet_(Dune::GeometryType(Dune::GeometryType::cube, GridPart::dimension - 1),
                   agIndexSet_.maxEdgeDegree()),
                polOrder_(polOrder),
                useOnb_(basisChoice == 2)
            {
#if 0
              const int innerTestSpace = testSpaces[2];
              assert(innerTestSpace>=-1);
              const std::size_t numShapeFunctions = scalarShapeFunctionSet_.size(); // uses polOrder
              // const std::size_t  //! this casuses a weird internal compiler error...
              int numGradShapeFunctions =
                      Dune::Fem::OrthonormalShapeFunctions< DomainType::dimension >::size(innerTestSpace+1);
              const std::size_t numInnerShapeFunctions = innerTestSpace<0?0:
                      Dune::Fem::OrthonormalShapeFunctions< DomainType::dimension >::size(innerTestSpace);
              std::cout << "******************************************\n";
              std::cout << "AgglomerationVEMSpace: "
                << polOrder << " (" << testSpaces[0] << "," << testSpaces[1] << "," << testSpaces[2] << ") "
                << "inner:" << numInnerShapeFunctions << "    "
                << "edge:" << edgeShapeFunctionSet_.size() << " " << "value:" << scalarShapeFunctionSet_.size() << " "
                << "grad:" << numGradShapeFunctions
                << std::endl;
              std::cout << "******************************************\n";
#endif
              onbBasis(agglomeration, scalarShapeFunctionSet_, agIndexSet_.boundingBox());
              buildProjections();
            }

            const BasisFunctionSetType basisFunctionSet(const EntityType &entity) const
            {
              const std::size_t agglomerate = agglomeration().index(entity);
              const auto &valueProjection = valueProjections_[agglomerate];
              const auto &jacobianProjection = jacobianProjections_[agglomerate];
              const auto &hessianProjection = hessianProjections_[agglomerate];
              const auto &bbox = blockMapper_.indexSet().boundingBox(agglomerate);
              // scalar ONB Basis proxy
              typename Traits::DGTraitsType::ShapeFunctionSetType scalarShapeFunctionSet(&scalarShapeFunctionSet_);
              // scalar BB Basis
              typename Traits::ScalarBBBasisFunctionSetType bbScalarBasisFunctionSet(entity, bbox,
                useOnb_, std::move(scalarShapeFunctionSet));
              // vectorial extended VEM Basis
              typename Traits::ScalarBasisFunctionSetType scalarBFS(entity, bbox, valueProjection, jacobianProjection, hessianProjection, std::move(bbScalarBasisFunctionSet));
              return BasisFunctionSetType(std::move(scalarBFS));
            }
            const typename Traits::ScalarBasisFunctionSetType scalarBasisFunctionSet(const EntityType &entity) const
            {
              const std::size_t agglomerate = agglomeration().index(entity);
              const auto &valueProjection = valueProjections_[agglomerate];
              const auto &jacobianProjection = jacobianProjections_[agglomerate];
              const auto &hessianProjection = hessianProjections_[agglomerate];
              const auto &bbox = blockMapper_.indexSet().boundingBox(agglomerate);
              // scalar ONB Basis proxy
              typename Traits::DGTraitsType::ShapeFunctionSetType scalarShapeFunctionSet(&scalarShapeFunctionSet_);
              // scalar BB Basis
              typename Traits::ScalarBBBasisFunctionSetType bbScalarBasisFunctionSet(entity, bbox,
                useOnb_, std::move(scalarShapeFunctionSet));
              // vectorial extended VEM Basis
              return typename Traits::ScalarBasisFunctionSetType(entity, bbox, valueProjection, jacobianProjection, hessianProjection, std::move(bbScalarBasisFunctionSet));
            }

            BlockMapperType &blockMapper() const { return blockMapper_; }

            // extra interface methods

            static constexpr bool continuous() noexcept { return false; }

            static constexpr bool continuous(const typename BaseType::IntersectionType &) noexcept { return false; }

            static constexpr int order(const EntityType &) noexcept { return polOrder; }
            static constexpr int order() { return polOrder; }

            static constexpr Fem::DFSpaceIdentifier type() noexcept { return Fem::GenericSpace_id; }

            // implementation-defined methods

            const AgglomerationType &agglomeration() const { return blockMapper_.agglomeration(); }

            const Stabilization &stabilization(const EntityType &entity) const { return stabilizations_[agglomeration().index(entity)]; }

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
#if 0
//            template<class Matrix>
//            struct ColumnVector {
//                ColumnVector(Matrix &matrix, int col)
//                        : matrix_(matrix), col_(col) {}
//
//                int size() const { return matrix_.size(); }
//
//                // typename Matrix::value_type& operator[](int i) {return matrix_[i][col_];}
//                template<class Vector>
//                ColumnVector &operator=(const Vector &v) {
//                  assert(v.size() == size());
//                  for (std::size_t i = 0; i < size(); ++i)
//                    matrix_[i][col_] = v[i];
//                  return *this;
//                }
//
//                Matrix &matrix_;
//                int col_;
//            };
//
//            template<class Matrix>
//            ColumnVector<Matrix> columnVector(Matrix &matrix, int col) { return ColumnVector<Matrix>(matrix, col); }
            // L.solve(d,b,columnVector(valueProjection,beta));
#endif

            void buildProjections();

            // issue with making these const: use of delete default constructor in some python bindings...
            AgglomerationIndexSetType agIndexSet_;
            mutable BlockMapperType blockMapper_;
            AgglomerationInterpolationType interpolation_;
            std::vector<typename Traits::ScalarBasisFunctionSetType::ValueProjection> valueProjections_;
            std::vector<typename Traits::ScalarBasisFunctionSetType::JacobianProjection> jacobianProjections_;
            std::vector<typename Traits::ScalarBasisFunctionSetType::HessianProjection> hessianProjections_;
            std::vector<Stabilization> stabilizations_;
            ScalarShapeFunctionSetType scalarShapeFunctionSet_;
            EdgeShapeFunctionSetType edgeShapeFunctionSet_;
            unsigned int polOrder_;
            const bool useOnb_;
        };


        // Implementation of AgglomerationVEMSpace
        // ---------------------------------------

        template<class FunctionSpace, class GridPart, int polOrder>
        inline void AgglomerationVEMSpace<FunctionSpace, GridPart, polOrder>::buildProjections()
        {
          // want to iterate over each polygon separately - so collect all
          // triangles from a given polygon
          std::vector<std::vector<ElementSeedType> > entitySeeds(agglomeration().size());
          for (const ElementType &element : elements(static_cast< typename GridPart::GridViewType >( gridPart()), Partitions::interiorBorder))
            entitySeeds[agglomeration().index(element)].push_back(element.seed());

          // get polynomial degree of inner test space and then compute
          // - size of space for value projection
          // - size of space for gradient projection
          // - size of space for inner moments
          //!TS add method to indexSet that returns correct orders as required, i.e.,
          //!TS   std::vector<int> orders = agIndexSet_.orders();
          //!TS Where orders.size() is always 2, i.e., for evaluate, jacobian, hessian
          std::vector<int> orders = agIndexSet_.orders();
//       const int innerTestSpace = agIndexSet_.testSpaces()[2];
//       assert(innerTestSpace>=-1);
//       innerTestSpace now orders[0]
//    // Question: how to choose the Grad/Hess spaces? Check Bubble and C1C0 space
          const std::size_t numShapeFunctions = scalarShapeFunctionSet_.size(); // uses polOrder
          int numHessShapeFunctions = std::min(numShapeFunctions,
                  Dune::Fem::OrthonormalShapeFunctions<DomainType::dimension>::
                  size(std::max(orders[2], polOrder - 2))
          );
          // polOrder=1 can cause problems with numHessShapeFunctions=0
          if (numHessShapeFunctions == 0)
            numHessShapeFunctions = 1;
          int numGradShapeFunctions = std::min(numShapeFunctions,
                  Dune::Fem::OrthonormalShapeFunctions<DomainType::dimension>::
                  size(std::max(orders[1], polOrder - 1))
          );
          const std::size_t numInnerShapeFunctions = orders[0] < 0 ? 0 :
                  Dune::Fem::OrthonormalShapeFunctions<DomainType::dimension>::
                  size(orders[0]);

#if 0
          numGradShapeFunctions = Dune::Fem::OrthonormalShapeFunctions< DomainType::dimension >::
                    size(polOrder-1);
          numHessShapeFunctions = Dune::Fem::OrthonormalShapeFunctions< DomainType::dimension >::
                    size(std::max(polOrder-2,0));
#endif

          const std::size_t numGradConstraints = Dune::Fem::OrthonormalShapeFunctions<DomainType::dimension>::
          size(orders[1]);
//            numGradShapeFunctions;




          std::cout << "size of spaces: "
                    << numInnerShapeFunctions << " "
                    << numShapeFunctions << " "
                    << numGradShapeFunctions << " "
                    << numHessShapeFunctions << " "
                    << numGradConstraints << std::endl;


          // set up matrices used for constructing gradient, value, and edge projections
          // Note: the code is set up with the assumption that the dofs suffice to compute the edge projection
          DynamicMatrix <DomainFieldType> D, C, constraintValueProj, constraintGradProj, leastSquaresGradProj, Hp, HpGrad, HpHess, HpInv, HpGradInv, HpHessInv;
          DynamicMatrix <DomainType> R; // ,G //!!!
          DynamicMatrix<typename Traits::ScalarBasisFunctionSetType::HessianMatrixType> P;

          LeftPseudoInverse <DomainFieldType> pseudoInverse(numShapeFunctions);

          // these are the matrices we need to compute
          valueProjections_.resize(agglomeration().size());
          jacobianProjections_.resize(agglomeration().size());
          hessianProjections_.resize(agglomeration().size());
          stabilizations_.resize(agglomeration().size());

          // start iteration over all polygons
          for (std::size_t agglomerate = 0; agglomerate < agglomeration().size(); ++agglomerate) {
            const auto &bbox = blockMapper_.indexSet().boundingBox(agglomerate);
            const std::size_t numDofs = blockMapper().numDofs(agglomerate);

            const int numEdges = agIndexSet_.subAgglomerates(agglomerate, AgglomerationIndexSetType::dimension - 1);
            const std::size_t edgeNormalSize = agIndexSet_.template order2size<1>(1);

            D.resize(numDofs, numShapeFunctions, 0);
            C.resize(numShapeFunctions, numDofs, 0);
            Hp.resize(numShapeFunctions, numShapeFunctions, 0);
            HpGrad.resize(numGradShapeFunctions, numGradShapeFunctions, 0);
            HpHess.resize(numHessShapeFunctions, numHessShapeFunctions, 0);
            //!!! G.resize( numGradShapeFunctions, numGradShapeFunctions, DomainType(0) );
            R.resize(numGradConstraints, numDofs, DomainType(0));
            P.resize(numHessShapeFunctions, numDofs, 0);
            constraintValueProj.resize(numInnerShapeFunctions, numShapeFunctions, 0);
            constraintGradProj.resize(numGradConstraints,numGradShapeFunctions,0);
            leastSquaresGradProj.resize( numEdges*edgeNormalSize , 2*numGradShapeFunctions,0);

            // iterate over the triangles of this polygon
            for (const ElementSeedType &entitySeed : entitySeeds[agglomerate]) {
              const ElementType &element = gridPart().entity(entitySeed);
              const auto geometry = element.geometry();
              const auto &refElement = ReferenceElements<typename GridPart::ctype, GridPart::dimension>::general(
                      element.type());

              // get the bounding box monomials and apply all dofs to them
              BoundingBoxBasisFunctionSet <GridPart, ScalarShapeFunctionSetType> shapeFunctionSet(element, bbox,
                      useOnb_, scalarShapeFunctionSet_);
              interpolation_(shapeFunctionSet, D);

              // compute mass matrices Hp, HpGrad, and the gradient matrices G^l
              Quadrature0Type quadrature(element, 2 * polOrder);
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
#if 0 // !!!!
                  if (alpha<numGradShapeFunctions)
                    shapeFunctionSet.jacobianEach( quadrature[ qp ], [ & ] ( std::size_t beta, typename ScalarShapeFunctionSetType::JacobianRangeType psi ) {
                      if (beta<numGradShapeFunctions)
                        G[alpha][beta].axpy(phi[0]*weight, psi[0]);
                    } );
#endif
                });
              } // quadrature loop
            } // loop over triangles in agglomerate

            // finished agglomerating all auxiliary matrices
            // now compute projection matrices and stabilization for the given polygon

            // volume of polygon
            DomainFieldType H0 = blockMapper_.indexSet().volume(agglomerate);

            auto &valueProjection = valueProjections_[agglomerate];
            auto &jacobianProjection = jacobianProjections_[agglomerate];
            auto &hessianProjection = hessianProjections_[agglomerate];
            valueProjection.resize(numShapeFunctions);
            jacobianProjection.resize(numShapeFunctions);
            hessianProjection.resize(numShapeFunctions);
            for (std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha) {
              valueProjection[alpha].resize(numDofs, 0);
              jacobianProjection[alpha].resize(numDofs, DomainType(0));
              hessianProjection[alpha].resize(numDofs, 0); // typename hessianProjection[alpha]::value_type( 0 ) );
            }

            // type def for standard vector (to pick up re size for Hessian projection)
            // need to resize Hessian projection

            // re implementation of the value projection
            auto leastSquaresMinimizer = LeastSquares(D, constraintValueProj);
            std::vector<DomainFieldType> b(numDofs, 0), d(numInnerShapeFunctions, 0);

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

            /////////////////////////////////////////
            /////////////////////////////////////////
// !!! Original value projection implementation
//        pseudoInverse( D, valueProjection );
            /////////////////////////////////////////
            /////////////////////////////////////////

            if (1 || numInnerShapeFunctions > 0) {
              // modify C for inner dofs
//          std::size_t alpha=0;
              /////////////////////////////////////////
              /////////////////////////////////////////
//// !!! Original value projection implementation
//          for (; alpha<numInnerShapeFunctions; ++alpha)
//            C[alpha][alpha+numDofs-numInnerShapeFunctions] = H0;
//          for (; alpha<numShapeFunctions; ++alpha)
//            for (std::size_t i=0; i<numDofs; ++i)
//              for (std::size_t beta=0; beta<numShapeFunctions; ++beta)
//                C[alpha][i] += Hp[alpha][beta]*valueProjection[beta][i];
              /////////////////////////////////////////
              /////////////////////////////////////////

              HpInv = Hp;
              HpInv.invert();
              HpGradInv = HpGrad;
              HpGradInv.invert();
              HpHessInv = HpHess;
              HpHessInv.invert();

#if 0 //!!!
              auto Gtmp = G;
              for (std::size_t alpha=0; alpha<numGradShapeFunctions; ++alpha)
              {
                for (std::size_t beta=0; beta<numGradShapeFunctions; ++beta)
                {
                  G[alpha][beta] = DomainType(0);
                  for (std::size_t gamma=0; gamma<numGradShapeFunctions; ++gamma)
                    G[alpha][beta].axpy(HpGradInv[alpha][gamma],Gtmp[gamma][beta]);
                }
              }

              { // test G matrix
                for( const ElementSeedType &entitySeed : entitySeeds[ agglomerate ] )
                {
                  const ElementType &element = gridPart().entity( entitySeed );
                  BoundingBoxBasisFunctionSet< GridPart, ScalarShapeFunctionSetType > shapeFunctionSet( element, bbox,
                      useOnb_, scalarShapeFunctionSet_ );
                  Quadrature0Type quad( element, 2*polOrder );
                  for( std::size_t qp = 0; qp < quad.nop(); ++qp )
                  {
                    shapeFunctionSet.jacobianEach( quad[qp], [ & ] ( std::size_t alpha, FieldMatrix< DomainFieldType, 1,2 > phi ) {
                      if (alpha<numGradShapeFunctions)
                      {
                        Dune::FieldVector<double,2> d;
                        Derivative<GridPartType, Dune::DynamicMatrix<DomainType>,decltype(shapeFunctionSet)>
                            derivative(gridPart(),G,shapeFunctionSet,alpha); // used G matrix to compute gradients of monomials
                        derivative.evaluate(quad[qp],d);
                        d -= phi[0];
                        assert(d.two_norm() < 1e-8);
                      }
                    });
                  }
                }
              } // test G matrix
#endif
#if 0
              // need to compute value projection first
               // now compute projection by multiplying with inverse mass matrix
              for (std::size_t alpha=0; alpha<numShapeFunctions; ++alpha)
              {
                for (std::size_t i=0; i<numDofs; ++i)
                {
                  valueProjection[alpha][i] = 0;
                  for (std::size_t beta=0; beta<numShapeFunctions; ++beta)
                    valueProjection[alpha][i] += HpInv[alpha][beta]*C[beta][i];
                  //!!!
                  // if (alpha<numGradShapeFunctions)
                    // for (std::size_t beta=0; beta<numGradShapeFunctions; ++beta)
                      // jacobianProjection[alpha][i].axpy(HpGradInv[alpha][beta],R[beta][i]);
                }
              }
#endif
              // // add interior integrals for gradient projection
              // for (std::size_t alpha=0; alpha<numGradShapeFunctions; ++alpha)
              //   for (std::size_t beta=0; beta<numInnerShapeFunctions; ++beta)
              //     R[alpha][beta+numDofs-numInnerShapeFunctions].axpy(-H0, G[beta][alpha]);

              /////////////////////////////////////////
              /////////////////////////////////////////
//// !!! Original value projection implementation
//          // now compute projection by multiplying with inverse mass matrix
//          for (std::size_t alpha=0; alpha<numShapeFunctions; ++alpha)
//          {
//            for (std::size_t i=0; i<numDofs; ++i)
//            {
//              valueProjection[alpha][i] = 0;
//              for (std::size_t beta=0; beta<numShapeFunctions; ++beta)
//                valueProjection[alpha][i] += HpInv[alpha][beta]*C[beta][i];
//              //!!!!
//              // if (alpha<numGradShapeFunctions)
//                // for (std::size_t beta=0; beta<numGradShapeFunctions; ++beta)
//                  // jacobianProjection[alpha][i].axpy(HpGradInv[alpha][beta],R[beta][i]);
//            }
//          }
              /////////////////////////////////////////
              /////////////////////////////////////////
            } // have some inner moments


            //////////////////////////////////////////////////////////////////////////
            /// GradientProjecjtion //////////////////////////////////////////////////
            //////////////////////////////////////////////////////////////////////////

            std::vector< std::vector<int>> fullMask;

            // iterate over the triangles of this polygon
            for (const ElementSeedType &entitySeed : entitySeeds[agglomerate]) {
              const ElementType &element = gridPart().entity(entitySeed);
              const auto geometry = element.geometry();
              const auto &refElement = ReferenceElements<typename GridPart::ctype, GridPart::dimension>::general(
                      element.type());
              std::vector<Dune::DynamicMatrix<double> > edgePhiVector(2);
              // get the bounding box monomials and apply all dofs to them
              BoundingBoxBasisFunctionSet <GridPart, ScalarShapeFunctionSetType> shapeFunctionSet(element, bbox,
                                                                                                  useOnb_,
                                                                                                  scalarShapeFunctionSet_);

              int counter = 0;

              // compute the boundary terms for the gradient projection
              for (const auto &intersection : intersections(static_cast< typename GridPart::GridViewType >( gridPart()),
                                                            element)) {
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
                  for (std::size_t j = 0; j < factorTN.cols; ++j) {
                    factorTN[i][j] = 0.5 * (normal[i] * tau[j] + normal[j] * tau[i]);
                    factorNN[i][j] = 0.5 * (normal[i] * normal[j] + normal[j] * normal[i]);
                  }

                std::vector<std::vector<int>> mask(2); // contains indices with Phi_mask[i] is attached to given edge
                edgePhiVector[0].resize(agIndexSet_.edgeSize(0), agIndexSet_.edgeSize(0), 0);
                edgePhiVector[1].resize(agIndexSet_.edgeSize(1), agIndexSet_.edgeSize(1), 0);
                interpolation_(intersection, edgeShapeFunctionSet_, edgePhiVector, mask);
                assert(mask[0].size() == agIndexSet_.edgeSize(0));
                assert(mask[1].size() == agIndexSet_.edgeSize(1));
                if (edgePhiVector[0].size() > 0)
                  edgePhiVector[0].invert();
                if (edgePhiVector[1].size() > 0)
                  edgePhiVector[1].invert();

                /* WARNING WARNING WARNING
                 * This is a horrible HACK and needs to be revised:
                 * It might be necessary to flip the vertex entries in the masks around
                 * due to a twist in the intersection.
                 * At the moment this is done by checking that the
                 * interpolation properties for phiEdge is satisfied - if not
                 * the mask is switch and the test is repeated!
                 * WARNING WARNING WARNING */
                {
                  std::vector<double> lambda(numDofs);
                  bool succ = true;
                  for (int i = 0; i < mask[0].size(); ++i) {
                    std::fill(lambda.begin(), lambda.end(), 0);
                    PhiEdge <GridPartType, Dune::DynamicMatrix<double>, EdgeShapeFunctionSetType>
                            phiEdge(gridPart(), intersection, edgePhiVector[0], edgeShapeFunctionSet_,
                                    i); // behaves like Phi_mask[i] restricted to edge
                    interpolation_(element, phiEdge, lambda);
                    for (int k = 0; k < numDofs; ++k) // lambda should be 1 for k=mask[i] otherwise 0
                      succ &= (mask[0][i] == k ? std::abs(lambda[k] - 1) < 1e-10 : std::abs(lambda[k]) < 1e-10);
                  }
                  if (!succ) std::swap(mask[0][0], mask[0][1]); // the HACK
                  succ = true;
                  for (int i = 0; i < mask[0].size(); ++i) {
                    std::fill(lambda.begin(), lambda.end(), 0);
                    PhiEdge <GridPartType, Dune::DynamicMatrix<double>, EdgeShapeFunctionSetType>
                            phiEdge(gridPart(), intersection, edgePhiVector[0], edgeShapeFunctionSet_,
                                    i); // behaves like Phi_mask[i] restricted to edge
                    interpolation_(element, phiEdge, lambda);
                    for (int k = 0; k < numDofs; ++k) // lambda should be 1 for k=mask[i] otherwise 0
                      succ &= (mask[0][i] == k ? std::abs(lambda[k] - 1) < 1e-10 : std::abs(lambda[k]) < 1e-10);
                  }
                  assert(succ);
                }

                // now compute int_e Phi_mask[i] m_alpha
                Quadrature1Type quadrature(gridPart(), intersection, 2 * polOrder, Quadrature1Type::INSIDE);
                for (std::size_t qp = 0; qp < quadrature.nop(); ++qp) {
                  auto x = quadrature.localPoint(qp);
                  auto y = intersection.geometryInInside().global(x);
                  const DomainFieldType weight = intersection.geometry().integrationElement(x) * quadrature.weight(qp);
                  shapeFunctionSet.evaluateEach(y, [&](std::size_t alpha, FieldVector<DomainFieldType, 1> phi) {
                      if (alpha < numGradShapeFunctions)
                        // evaluate each here for edge shape fns
                        edgeShapeFunctionSet_.evaluateEach(x, [&](std::size_t beta, FieldVector<DomainFieldType, 1> psi) {
                          if (beta < edgePhiVector[0].size() && alpha < numGradConstraints ) {
                            for (int s = 0; s <
                                            mask[0].size(); ++s)// note that edgePhi is the transposed of the basis transform matrix
                                R[alpha][mask[0][s]].axpy(edgePhiVector[0][beta][s] * psi[0] * phi[0] * weight, normal);
                            //assemble left hand side here for ls problem
                          }
                          if (beta < edgePhiVector[1].size()) {
                            leastSquaresGradProj[counter+beta][alpha] += psi[0] * phi[0] * weight * normal[0];
                            leastSquaresGradProj[counter+beta][alpha + numGradShapeFunctions ] += psi[0] * phi[0] * weight * normal[1];
                          }
                        });
                      if (alpha < numHessShapeFunctions) // && agIndexSet_.edgeSize(1) > 0)
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
                              for (int s = 0; s <
                                              mask[0].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                                P[alpha][mask[0][s]].axpy(edgePhiVector[0][beta][s] * gradPsiDottau * phi[0] * weight,
                                                          factorTN);
                            }
                        });
#endif
                      } // alpha < numHessSF
                      if (alpha < numHessShapeFunctions && agIndexSet_.edgeSize(1) > 0) {
                        edgeShapeFunctionSet_.evaluateEach(x, [&](std::size_t beta, FieldVector<DomainFieldType, 1> psi) {
                          if (beta < edgePhiVector[1].size())
                            for (int s = 0; s < mask[1].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                              P[alpha][mask[1][s]].axpy(edgePhiVector[1][beta][s] * psi[0] * phi[0] * weight, factorNN);
                        });
                      } // alpha < numHessSF and can compute normal derivative
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
                counter += agIndexSet_.edgeSize(1);
              } // loop over intersections


              auto vemBasisFunction = scalarBasisFunctionSet(element);
              Quadrature0Type quadrature(element, 2 * polOrder);
              for (std::size_t qp = 0; qp < quadrature.nop(); ++qp) {
                const DomainFieldType weight =
                        geometry.integrationElement(quadrature.point(qp)) * quadrature.weight(qp);
                shapeFunctionSet.jacobianEach(quadrature[qp], [&](std::size_t alpha,
                                                                  typename ScalarShapeFunctionSetType::JacobianRangeType gradPhi) {
                    // Note: the shapeFunctionSet is defined in physical space so
                    // the jit is not needed here
                    // R[alpha][j]  -=  Pi phi_j  grad(m_alpha) * weight
                    if (alpha < numGradConstraints) {
                      gradPhi[0] *= -weight;
                      vemBasisFunction.axpy(quadrature[qp], gradPhi[0], R[alpha]);
                    }
                });

              } // quadrature loop



              ////////////////////////////////////////////////////////////
              //// re-implementation of gradient projection //////////////
              ////////////////////////////////////////////////////////////

              // set up RHS of least squares
              // int_e nabla V ph_j dot n m_beta


//              std::vector<DomainFieldType > b(rowSizeLeastSquares,0);
//
//              Quadrature1Type quadrature(gridPart(), intersection, 2 * polOrder, Quadrature1Type::INSIDE);
//              for (std::size_t qp = 0; qp < quadrature.nop(); ++qp) {
//                auto x = quadrature.localPoint(qp);
//                auto y = intersection.geometryInInside().global(x);
//                const DomainFieldType weight = intersection.geometry().integrationElement(x) * quadrature.weight(qp);
//                shapeFunctionSet.jacobianEach(y, [&](std::size_t alpha, typename ScalarShapeFunctionSetType::JacobianRangeType gradPhi) {
//                  if (alpha < numGradShapeFunctions)
//                    // evaluate each here for edge shape fns
//                    edgeShapeFunctionSet_.evaluateEach(x, [&](std::size_t beta, FieldVector<DomainFieldType, 1> psi) {
//                      if (beta < edgePhiVector[0].size()) //assemble left hand side here for ls problem
//                        b[beta] += valueProjection[alpha][beta] * psi[0] * ( gradPhi[0] * normal[0] + gradPhi[1] * normal[1] );
//                    });
//                });
//              }

              // set up R and enlarged constraint matrix

              ////////////////////////////////////////////////////////////
              ///////////////////////////////////////////////////////////



            } // loop over triangles in agglomerate


            // least squares
            if ( leastSquaresGradProj.size() == 0 ) {
              if (1 || numInnerShapeFunctions > 0) {

                assert( numGradConstraints == numGradShapeFunctions );

                // need to compute value projection first
                // now compute projection by multiplying with inverse mass matrix
                for (std::size_t alpha = 0; alpha < numGradShapeFunctions; ++alpha)
                  for (std::size_t i = 0; i < numDofs; ++i)
                    for (std::size_t beta = 0; beta < numGradShapeFunctions; ++beta)
                      jacobianProjection[alpha][i].axpy(HpGradInv[alpha][beta], R[beta][i]);
              } // have some inner moments
              else { // with no inner moments we didn't need to compute the inverse of the mass matrix H_{p-1} -
                // it's simply 1/H0 so we implement this case separately
                for (std::size_t i = 0; i < numDofs; ++i)
                  jacobianProjection[0][i].axpy(1. / H0, R[0][i]);
              }
            }
            else {
              BlockMatrix constraintBlockMatrix = blockMatrix(constraintGradProj, 2);
              auto leastSquaresMinimizerGradient = leastSquares(leastSquaresGradProj, constraintBlockMatrix);
              std::vector<DomainFieldType> b(leastSquaresGradProj.rows(), 0);

              std::cout << "b size " << b.size() << std::endl;
              std::cout << "num dofs " << numDofs << std::endl;

              for (std::size_t beta = 0; beta < numDofs; ++beta) {

                std::cout << " beta: " << beta << std::endl;
                std::cout << " R size: " << R.size() << std::endl;
                std::cout << " grad projection size: " << jacobianProjection.size() << std::endl;

                VectorizeMatrixCol d = vectorizeMatrixCol(R, beta);
                VectorizeMatrixCol colGradProjection = vectorizeMatrixCol(jacobianProjection, beta);

                int counter = 0;
                bool finished = false;
                for (int e = 0; e < fullMask.size(); e++)
                {
                  finished = false;
                  std::cout << " e: " << e << std::endl;
                  for (int i = 0; i < fullMask[e].size(); i++, ++counter)
                  {
                    std::cout << " i: " << i << std::endl;
                    if (fullMask[e][i] == beta) {
                      std::cout << "beta " << beta << " fullmask " << fullMask[e][i] << std::endl;
                      b[counter] = 1;
                      finished = true;
                      break;
                    }
                  }
                  if (finished) break;
                }
                std::cout << " counter: " << counter << std::endl;

                colGradProjection = leastSquaresMinimizerGradient.solve(b, d);

                std::cout << " I reached here " << std::endl;

                // re set b vector to 0
                if (finished)
                  b[counter] = 0;
              }
            }

            /////////////////////////////////////////////////////////////////////
            // HessianProjection ////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////

            // iterate over the triangles of this polygon (for Hessian projection)
            for (const ElementSeedType &entitySeed : entitySeeds[agglomerate]) {
              const ElementType &element = gridPart().entity(entitySeed);
              const auto geometry = element.geometry();
              const auto &refElement = ReferenceElements<typename GridPart::ctype, GridPart::dimension>::general(
                      element.type());

              // get the bounding box monomials and apply all dofs to them
              BoundingBoxBasisFunctionSet <GridPart, ScalarShapeFunctionSetType> shapeFunctionSet(element, bbox,
                      useOnb_, scalarShapeFunctionSet_);
              auto vemBasisFunction = scalarBasisFunctionSet(element);

              // compute the boundary terms for the gradient projection
              if (agIndexSet_.edgeSize(1) == 0) {
                for (const auto &intersection : intersections(
                        static_cast< typename GridPart::GridViewType >( gridPart()), element)) {
                  // ignore edges inside the given polygon
                  if (!intersection.boundary() && (agglomeration().index(intersection.outside()) == agglomerate))
                    continue;
                  assert(intersection.conforming());
                  auto normal = intersection.centerUnitOuterNormal();

                  // change to compute boundary term in Hessian Projection
                  // now compute int_e Phi_mask[i] m_alpha
                  Quadrature1Type quadrature(gridPart(), intersection, 2 * polOrder, Quadrature1Type::INSIDE);
                  for (std::size_t qp = 0; qp < quadrature.nop(); ++qp) {
                    auto x = quadrature.localPoint(qp);
                    auto y = intersection.geometryInInside().global(x);
                    const DomainFieldType weight =
                            intersection.geometry().integrationElement(x) * quadrature.weight(qp);
                    shapeFunctionSet.evaluateEach(y, [&](std::size_t alpha, FieldVector<DomainFieldType, 1> phi) {
                        if (alpha < numHessShapeFunctions) {
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
              Quadrature0Type quadrature(element, 2 * polOrder);
              for (std::size_t qp = 0; qp < quadrature.nop(); ++qp) {
                const DomainFieldType weight =
                        geometry.integrationElement(quadrature.point(qp)) * quadrature.weight(qp);
                shapeFunctionSet.jacobianEach(quadrature[qp], [&](std::size_t alpha,
                                                                  typename ScalarShapeFunctionSetType::JacobianRangeType gradPhi) {
                    // Note: the shapeFunctionSet is defined in physical space so
                    // the jit is not needed here
                    // P[alpha][j] -= Pi grad phi_j grad(m_alpha) * weight
                    if (alpha < numHessShapeFunctions) {
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
                for (std::size_t beta = 0; beta < numHessShapeFunctions; ++beta)
                  hessianProjection[alpha][i].axpy(HpHessInv[alpha][beta], P[beta][i]);

            /////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////
#if 0
            // compute energy norm stability scalling
            std::vector<double> stabScaling(numDofs, 0);
            std::vector<typename BasisFunctionSetType::JacobianRangeType> dphi(numDofs);
            for( const ElementSeedType &entitySeed : entitySeeds[ agglomerate ] )
            {
              const ElementType &element = gridPart().entity( entitySeed );
              const auto geometry = element.geometry();
              const BasisFunctionSetType vemBaseSet = basisFunctionSet ( element );
              Fem::ElementQuadrature< GridPart, 0 > quadrature( element, 2*polOrder );
              for( std::size_t qp = 0; qp < quadrature.nop(); ++qp )
              {
                vemBaseSet.jacobianAll(quadrature[qp], dphi);
                const auto &x = quadrature.point(qp);
                const double weight = quadrature.weight(qp) * geometry.integrationElement(x);
                for (int i=0;i<numDofs;++i)
                  stabScaling[i] += weight*(dphi[i][0]*dphi[i][0]);
              }
            }
            std::cout << "stabscaling= " << std::flush;
            for( std::size_t i = 0; i < numDofs; ++i )
              std::cout << stabScaling[i] << " ";
            std::cout << std::endl;
#endif
#if 1
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
            Stabilization &stabilization = stabilizations_[agglomerate];
            stabilization.resize(numDofs, numDofs, 0);
            for (std::size_t i = 0; i < numDofs; ++i)
              for (std::size_t j = 0; j < numDofs; ++j) {
                for (std::size_t k = 0; k < numDofs; ++k)
                  stabilization[i][j] += S[k][i] * S[k][j];
                // stabilization[ i ][ j ] += S[ k ][ i ] * std::max(1.,stabScaling[k]) * S[ k ][ j ];
                // std::cout << "   " << i << " " << j << " "
                //   << stabilization[i][j] << " " << S[i][j] << std::endl;
              }
#else
            Stabilization &stabilization = stabilizations_[ agglomerate ];
            stabilization.resize( numDofs, numDofs, 0 );
            for( std::size_t i = 0; i < numDofs; ++i )
              stabilization[ i ][ i ] = DomainFieldType( 1 );
            for( std::size_t i = 0; i < numDofs; ++i )
              for( std::size_t alpha = 0; alpha < numShapeFunctions; ++alpha )
                for( std::size_t j = 0; j < numDofs; ++j )
                  stabilization[ i ][ j ] -= D[ i ][ alpha ] * valueProjection[ alpha ][ j ];
#endif

            //compute R using value projection
            //how to access grad m_\alpha, need to use G
            //how to call evaluate all


            // Need to implement hessian projections
            // Need
            // quad rule for inner and edges
            // jacobianall
            // (only one IBP)

          } // iterate over polygons

        } // build projections

    } // namespace Vem



    namespace Fem {

        // External Forward Declarations
        // -----------------------------

#if HAVE_DUNE_ISTL
        template< class Matrix, class Space >
        struct ISTLParallelMatrixAdapter;

        template< class Matrix >
        class LagrangeParallelMatrixAdapter;
#endif // #if HAVE_DUNE_ISTL



        // ISTLParallelMatrixAdapter for AgglomerationVEMSpace
        // ---------------------------------------------------

#if HAVE_DUNE_ISTL
        template< class Matrix, class FunctionSpace, class GridPart, int polOrder >
        struct ISTLParallelMatrixAdapter< Matrix, Vem::AgglomerationVEMSpace< FunctionSpace, GridPart, polOrder > >
        {
          typedef LagrangeParallelMatrixAdapter< Matrix > Type;
        };
#endif // #if HAVE_DUNE_ISTL

        namespace Capabilities {
            template<class FunctionSpace, class GridPart, int polOrder>
            struct hasInterpolation<Vem::AgglomerationVEMSpace<FunctionSpace, GridPart, polOrder> > {
                static const bool v = false;
            };
        }

    } // namespace Fem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_AGGLOMERATION_HH
