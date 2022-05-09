#ifndef DUNE_VEM_SPACE_CURLFREE_HH
#define DUNE_VEM_SPACE_CURLFREE_HH

#include <cassert>
#include <utility>

#include <dune/common/dynmatrix.hh>
#include <dune/geometry/referenceelements.hh>
#include <dune/fem/quadrature/elementquadrature.hh>
#include <dune/fem/space/common/defaultcommhandler.hh>
#include <dune/fem/space/common/discretefunctionspace.hh>
#include <dune/fem/space/common/functionspace.hh>
#include <dune/fem/space/common/capabilities.hh>
#include <dune/fem/space/shapefunctionset/orthonormal.hh>
#include <dune/fem/function/localfunction/converter.hh>
#include <dune/fem/space/combinedspace/interpolation.hh>
#include <dune/vem/misc/compatibility.hh>

#include <dune/vem/agglomeration/basisfunctionset.hh>
#include <dune/vem/misc/vector.hh>
#include <dune/vem/space/default.hh>
#include <dune/vem/space/hk.hh>

namespace Dune
{
  namespace Vem
  {
    // Internal Forward Declarations
    // -----------------------------

    template<class GridPart>
    class CurlFreeVEMSpace;

    template<class GridPart>
    struct IsAgglomerationVEMSpace<CurlFreeVEMSpace<GridPart> >
            : std::integral_constant<bool, true> {
    };

    // CurlFreeVEMSpaceTraits
    // ---------------------------

    template<class FunctionSpace, class GridPart, bool reduced=true>
    struct CurlFreeVEMBasisSets
    {
      typedef GridPart GridPartType;
      static constexpr int dimDomain = GridPartType::dimension;
      typedef typename GridPart::template Codim<0>::EntityType EntityType;
      typedef typename GridPart::IntersectionType IntersectionType;

      typedef Dune::Fem::FunctionSpace<
              typename FunctionSpace::DomainFieldType, typename FunctionSpace::RangeFieldType,
              dimDomain, 1 > ScalarFunctionSpaceType;
      typedef Dune::Fem::OrthonormalShapeFunctionSet< ScalarFunctionSpaceType > ONBShapeFunctionSetType;
      typedef BoundingBoxBasisFunctionSet< GridPartType, ONBShapeFunctionSetType > ScalarBBBasisFunctionSetType;

      // Next we define test function space for the edges
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld-1,1> EdgeFSType;
      typedef Dune::Fem::OrthonormalShapeFunctionSet<EdgeFSType> ScalarEdgeShapeFunctionSetType;

    private:
      struct ShapeFunctionSet
      {
        typedef typename ScalarFunctionSpaceType::RangeType ScalarRangeType;
        typedef typename ScalarFunctionSpaceType::JacobianRangeType ScalarJacobianRangeType;
        typedef typename ScalarFunctionSpaceType::HessianRangeType ScalarHessianRangeType;

        typedef FunctionSpace FunctionSpaceType;
        typedef typename FunctionSpaceType::RangeType RangeType;
        typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;
        typedef typename FunctionSpaceType::HessianRangeType HessianRangeType;
        static constexpr int dimDomain = FunctionSpaceType::DomainType::dimension;
        static constexpr int dimRange = RangeType::dimension;

        static_assert(dimRange==dimDomain);

        ShapeFunctionSet() = default;
        template <class Agglomeration>
        ShapeFunctionSet(bool useOnb, const ONBShapeFunctionSetType &onbSFS,
                         std::size_t numValueSFS, std::size_t numGradSFS, std::size_t numHessSFS,
                         std::size_t innerNumSFS,
                         const Agglomeration &agglomeration, const EntityType &entity)
        : sfs_(entity, agglomeration.index(entity),
               agglomeration.boundingBoxes(), useOnb, onbSFS)
        , scale_( std::sqrt( sfs_.bbox().volume() ) )
        , numValueShapeFunctions_(numValueSFS)
        , numGradShapeFunctions_(numGradSFS)
        , numHessShapeFunctions_(numHessSFS)
        , numInnerShapeFunctions_(innerNumSFS)
        {}

        int order () const { return sfs_.order()-1;  }

        // Note: jacobianEach not needed for interpolation so return 'this' works here
        const auto &valueBasisSet() const
        {
          return *this;
        }

        template <int codim>
        int vectordofs() const
        {
          return 1;
        }
        // Note: needed for the finalizing projection
        //       Could make this 'protected' and make the CurlFreeVemSpace 'friend'
        template< class Point, class Functor >
        void scalarEach ( const Point &x, Functor functor ) const
        {
          sfs_.evaluateEach(x, [&](std::size_t alpha, ScalarRangeType phi)
          {
            if (alpha>=1)
            {
              // phi[0] *= scale_;
              functor(alpha-1, phi[0]);
            }
          });
        }
        template< class Point, class Functor >
        void evaluateEach ( const Point &x, Functor functor ) const
        {
          sfs_.jacobianEach(x, [&](std::size_t alpha, ScalarJacobianRangeType dphi)
          {
            if (alpha>=1)
            {
              // dphi[0] *= scale_;
              functor(alpha-1, dphi[0]);
            }
          });
        }
        /*
          int Gu : M = int Du : M = - int u . divM + int_e uxn : M
          use M = mI so int Gu : M = int div u m
        */
        template< class Point, class Functor >
        void jacobianEach ( const Point &x, Functor functor ) const
        {
          sfs_.evaluateEach(x, [&](std::size_t alpha, ScalarRangeType phi)
          {
            if (alpha<numGradShapeFunctions_)
            {
              functor(alpha,{{phi[0],0},{0,phi[0]}});
            /*
              functor(2*alpha,   {{phi[0],0}, {0,0}});
              functor(2*alpha+1, {{0,0},      {0,phi[0]}});
            */
            }
          });
        }
        template< class Point, class Functor >
        void hessianEach ( const Point &x, Functor functor ) const
        {}
        // functor(alpha, psi) with psi in R^r
        //
        // for each g = g_{alpha*dimDomain+s} = m_alpha e_s    (1<=alpha<=numGradSF and 1<=s<=dimDomain)
        // sum_{ij} int_E d_j v_i g_ij = - sum_{ij} int_E v_i d_j g_ij + ...
        //     = - int_E sum_i ( v_i sum_j d_j g_ij )
        //     = - int_E v . psi
        // with psi_i = sum_j d_j g_ij
        //
        // g_{ij} = m_i delta_{js}   (m=m_alpha and fixed s=1,..,dimDomain)
        // psi_i = sum_j d_j g_ij = sum_j d_j m_i delta_{js} = d_s m_i
        template< class Point, class Functor >
        void divJacobianEach( const Point &x, Functor functor ) const
        {
          sfs_.jacobianEach(x, [&](std::size_t alpha, ScalarJacobianRangeType dphi)
          {
            if (alpha<numGradShapeFunctions_)
            {
              functor(alpha,dphi[0]);
              // functor(2*alpha,   {dphi[0][0],0});
              // functor(2*alpha+1, {0,dphi[0][1]});
            }
          });
        }
        template< class Point, class Functor >
        void divHessianEach( const Point &x, Functor functor ) const
        {
        }
        template< class Point, class Functor >
        void evaluateTestEach ( const Point &xx, Functor functor ) const
        {
          sfs_.jacobianEach(xx, [&](std::size_t alpha, ScalarJacobianRangeType dphi)
          {
            if (alpha>=1 && alpha < numInnerShapeFunctions_+1)
            {
              dphi[0] *= scale_;
              functor(alpha-1, dphi[0]);
            }
          });
        }

        private:
        ScalarBBBasisFunctionSetType sfs_;
        double scale_;
        std::size_t numValueShapeFunctions_;
        std::size_t numGradShapeFunctions_;
        std::size_t numHessShapeFunctions_;
        std::size_t innerShapeFunctions_;
        std::size_t numInnerShapeFunctions_;
      };

      struct EdgeShapeFunctionSet
      {
        typedef FunctionSpace FunctionSpaceType;
        typedef typename FunctionSpaceType::DomainType DomainType;
        typedef typename FunctionSpaceType::RangeType RangeType;
        typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;
        typedef typename FunctionSpaceType::HessianRangeType HessianRangeType;
        static constexpr int dimDomain = DomainType::dimension;
        static constexpr int dimRange = RangeType::dimension;

        typedef typename ScalarEdgeShapeFunctionSetType::RangeType EdgeRangeType;
        typedef typename ScalarEdgeShapeFunctionSetType::JacobianRangeType EdgeJacobianRangeType;

        EdgeShapeFunctionSet(const IntersectionType &intersection, int flip,
                             const ScalarEdgeShapeFunctionSetType &sfs)
        : intersection_(intersection), flip_(flip), sfs_(sfs)
        {}
        template< class Point, class Functor >
        void evaluateEach ( const Point &x, Functor functor ) const
        {
          // phihat = (1)
          //    phi = (d) = phihat normal
          sfs_.evaluateEach(x, [&](std::size_t alpha, EdgeRangeType phihat)
          {
            RangeType phi = intersection_.unitOuterNormal(x);
            phi *= phihat[0]*flip_;
            functor(alpha, phi);
          });
        }
        template< class Point, class Functor >
        void jacobianEach ( const Point &x, Functor functor ) const
        {
          assert(0);
        }
        unsigned int size() const
        {
          return sfs_.size();
        }
        template< class Point, class Functor >
        void evaluateTestEach ( const Point &x, Functor functor ) const
        {
          evaluateEach(x, functor);
        }
        private:
        const IntersectionType &intersection_;
        const int flip_;
        ScalarEdgeShapeFunctionSetType sfs_;
      };

      public:
      typedef ShapeFunctionSet ShapeFunctionSetType;
      typedef EdgeShapeFunctionSet EdgeShapeFunctionSetType;

      CurlFreeVEMBasisSets( const int order, bool useOnb)
      : innerOrder_( order )
      , onbSFS_( Dune::GeometryType(Dune::GeometryType::cube, dimDomain), order+1 )
      , edgeSFS_( Dune::GeometryType(Dune::GeometryType::cube,dimDomain-1), order)
      , dofsPerCodim_( calcDofsPerCodim(order) )
      , useOnb_(useOnb)
      , numValueShapeFunctions_( sizeONB<0>(order+1)-1 )
      , numGradShapeFunctions_ ( sizeONB<0>(order) )
      , numHessShapeFunctions_ ( 0 ) // not implemented - needs third/forth derivatives
      , numInnerShapeFunctions_( sizeONB<0>(innerOrder_)-1 )
      , numEdgeTestShapeFunctions_( edgeSFS_.size() )
      {
        std::cout << "[" << numValueShapeFunctions_ << ","
                  << numGradShapeFunctions_ << ","
                  << numHessShapeFunctions_ << ","
                  << numInnerShapeFunctions_ << "]"
                  << "   edge: " << edgeSFS_.size()
                  << std::endl;
        std::cout << "dofs per codim: "
                  << dofsPerCodim_[0].second << " " << dofsPerCodim_[1].second << " " << dofsPerCodim_[2].second
                  << std::endl;
      }

      const std::array< std::pair< int, unsigned int >, dimDomain+1 > &dofsPerCodim() const
      {
        return dofsPerCodim_;
      }

      template <class Agglomeration>
      ShapeFunctionSetType basisFunctionSet(
             const Agglomeration &agglomeration, const EntityType &entity) const
      {
        return ShapeFunctionSet(useOnb_, onbSFS_,
                                numValueShapeFunctions_, numGradShapeFunctions_, numHessShapeFunctions_,
                                numInnerShapeFunctions_,
                                agglomeration,entity);
      }
      template <class Agglomeration>
      EdgeShapeFunctionSetType edgeBasisFunctionSet(
             const Agglomeration &agglomeration, const IntersectionType &intersection) const
      {
        int flip = 1;
        auto normal = intersection.centerUnitOuterNormal();
        assert( abs(normal[0] + std::sqrt(2.)*normal[1]) > 1e-5 );
        if (intersection.neighbor())
        {
          // !!! FIXME: if (indexSet_.index(intersection.inside()) > indexSet_.index(intersection.outside()))
          if (normal[0] + std::sqrt(2.)*normal[1] < 0)
            flip = -1;
        }
        return EdgeShapeFunctionSetType(intersection, flip, edgeSFS_);
      }
      std::size_t size( std::size_t orderSFS ) const
      {
        if (orderSFS == 0)
          return numValueShapeFunctions_;
        else if (orderSFS == 1)
          return numGradShapeFunctions_;
        else if (orderSFS == 2)
          return numHessShapeFunctions_;
        assert(0);
        return 0;
      }
      int constraintSize() const
      {
        return numValueShapeFunctions_;
      }
      int vertexSize(int deriv) const
      {
        // no vertex moments in curl free space
        return 0;
      }
      int innerSize() const
      {
        return numInnerShapeFunctions_;
      }
      int edgeValueMoments() const
      {
        // curl free case this is equal to order of ESFS
        // equal to entry of testSpaces[1][0] in hk case
        return edgeSFS_.order();
      }

      std::size_t edgeSize(int deriv) const
      {
        return (deriv==0)? edgeSFS_.size() : 0;
      }

      ///////////////////////////
      // used in interpolation //
      // modified from hk      //
      ///////////////////////////
      std::size_t edgeSize() const
      {
        // returns size of edge shape function set -> same as above method for curl free space
        // (as no vertex values or edge normal moments)
        return edgeSize(0);
      }
      std::size_t numEdgeTestShapeFunctions() const
      {
        return numEdgeTestShapeFunctions_;
      }
      template <int dim>
      std::size_t order2size(unsigned int deriv) const
      {
        // size of either edge value moments or inner moments
        // only return sizes in these two cases
        if (dim == 1 && deriv == 0)
          return edgeSize();
        if (dim == 2 && deriv == 0)
          return innerSize();
        else
          return 0;
      }
      int vectorDofs(int dim) const
      {
        // in all cases scalar dofs in curl free space
        return 1;
      }

      private:
      std::array< std::pair< int, unsigned int >, dimDomain+1 > calcDofsPerCodim (unsigned int order) const
      {
        int vSize = 0;
        int eSize = edgeSFS_.size();
        int iSize = sizeONB<0>(innerOrder_)-1;
        return std::array< std::pair< int, unsigned int >, dimDomain+1 >
               { std::make_pair( dimDomain,   vSize ),
                 std::make_pair( dimDomain-1, eSize ),
                 std::make_pair( dimDomain-2, iSize ) };
      }

      template <int codim>
      static std::size_t sizeONB(std::size_t order)
      {
        return Dune::Fem::OrthonormalShapeFunctions<dimDomain - codim> :: size(order);
      }
      // note: the actual shape function set depends on the entity so
      // we can only construct the underlying monomial basis in the ctor
      const int innerOrder_;
      const ONBShapeFunctionSetType onbSFS_;
      const ScalarEdgeShapeFunctionSetType edgeSFS_;
      const std::array< std::pair< int, unsigned int >, dimDomain+1 > dofsPerCodim_;
      const std::size_t numValueShapeFunctions_;
      const std::size_t numGradShapeFunctions_;
      const std::size_t numHessShapeFunctions_;
      const std::size_t numInnerShapeFunctions_;
      const std::size_t numEdgeTestShapeFunctions_;
      const bool useOnb_;
    };



    template<class GridPart>
    struct CurlFreeVEMSpaceTraits
    {
      typedef GridPart GridPartType;
      static const int dimension = GridPartType::dimension;
      typedef Dune::Fem::FunctionSpace<double,double,dimension,dimension> FunctionSpaceType;

      typedef CurlFreeVEMBasisSets<FunctionSpaceType,GridPart> BasisSetsType;
      friend class CurlFreeVEMSpace<GridPart>;
      typedef CurlFreeVEMSpace<GridPart> DiscreteFunctionSpaceType;

      static const int dimDomain = FunctionSpaceType::DomainType::dimension;
      static const int dimRange = FunctionSpaceType::RangeType::dimension;
      static const int codimension = 0;
      static const bool vectorSpace = true;

      typedef typename GridPartType::template Codim<0>::EntityType EntityType;

      // vem basis function sets
      typedef VEMBasisFunctionSet <EntityType, typename BasisSetsType::ShapeFunctionSetType> BasisFunctionSetType;
      typedef BasisFunctionSetType ScalarBasisFunctionSetType;

      // types for the mapper
      typedef Hybrid::IndexRange<int, 1> LocalBlockIndices;
      typedef VemAgglomerationIndexSet <GridPartType> IndexSetType;
      typedef AgglomerationDofMapper <GridPartType, IndexSetType> BlockMapperType;

      // replace vectorSpace with basisFunctionSet::RangeType::dimension == dimRange
      static const int baseBlockSize = vectorSpace ? LocalBlockIndices::size() : 1;

      template<class DiscreteFunction, class Operation = Fem::DFCommunicationOperation::Copy>
      struct CommDataHandle {
          typedef Operation OperationType;
          typedef Fem::DefaultCommunicationHandler <DiscreteFunction, Operation> Type;
      };

      template <class T>
      using InterpolationType = AgglomerationVEMInterpolation<T>;
    };

    // CurlFreeVEMSpace
    // ---------------------
    template<class GridPart>
    struct CurlFreeVEMSpace
    : public DefaultAgglomerationVEMSpace< CurlFreeVEMSpaceTraits<GridPart> >
    {
      typedef CurlFreeVEMSpaceTraits<GridPart> TraitsType;
      typedef DefaultAgglomerationVEMSpace< TraitsType > BaseType;
      typedef typename BaseType::AgglomerationType AgglomerationType;
      typedef typename BaseType::BasisSetsType::ShapeFunctionSetType::FunctionSpaceType FunctionSpaceType;
      typedef typename FunctionSpaceType::DomainFieldType DomainFieldType;
      CurlFreeVEMSpace(AgglomerationType &agglomeration,
          const unsigned int polOrder,
          int basisChoice)
      : BaseType(agglomeration, polOrder,
                 typename TraitsType::BasisSetsType(polOrder, basisChoice==2),
                 basisChoice, false)
      {
        if (basisChoice != 3) // !!!!! get order information from BasisSets
          BaseType::agglomeration().onbBasis(polOrder+1);
        BaseType::update(true);
      }

    protected:
      virtual void setupConstraintRHS(const Std::vector<Std::vector<typename BaseType::ElementSeedType> > &entitySeeds, unsigned int agglomerate,
                                    Dune::DynamicMatrix<DomainFieldType> &RHSconstraintsMatrix, double volume) override
      {
        //////////////////////////////////////////////////////////////////////////
        /// Fix RHS constraints for value projection /////////////////////////////
        //////////////////////////////////////////////////////////////////////////

        typedef typename BaseType::BasisSetsType::EdgeShapeFunctionSetType EdgeTestSpace;
        typedef typename FunctionSpaceType::DomainType DomainType;
        typedef typename FunctionSpaceType::RangeFieldType RangeFieldType;
        typedef typename FunctionSpaceType::RangeType RangeType;
        typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;
        typedef typename FunctionSpaceType::HessianRangeType HessianRangeType;
        static constexpr int blockSize = BaseType::localBlockSize;
        const std::size_t dimDomain = DomainType::dimension;
        const std::size_t dimRange = RangeType::dimension;
        const std::size_t numShapeFunctions = BaseType::basisSets_.size(0);
        const std::size_t numDofs = BaseType::blockMapper().numDofs(agglomerate) * blockSize;
        int polOrder = BaseType::order();
        const std::size_t numConstraintShapeFunctions = BaseType::basisSets_.constraintSize();
        const std::size_t numInnerShapeFunctions = BaseType::basisSets_.innerSize();

        assert( numInnerShapeFunctions <= numConstraintShapeFunctions );
        if (numConstraintShapeFunctions == 0) return;

        // first fill in entries relating to inner dofs (alpha < inner shape functions)
        for ( int beta=0; beta<numDofs; ++beta)
        {
          for (int alpha=0; alpha<numInnerShapeFunctions; ++alpha)
          {
            if( beta - numDofs + numInnerShapeFunctions == alpha )
              RHSconstraintsMatrix[ beta ][ alpha ] = std::sqrt(volume);
          }
        }

        // matrices for edge projections
        Std::vector<Dune::DynamicMatrix<double> > edgePhiVector(2);
        edgePhiVector[0].resize(BaseType::basisSets_.edgeSize(0), BaseType::basisSets_.edgeSize(0), 0);
        edgePhiVector[1].resize(BaseType::basisSets_.edgeSize(1), BaseType::basisSets_.edgeSize(1), 0);

        for (const typename BaseType::ElementSeedType &entitySeed : entitySeeds[agglomerate])
        {
          const typename BaseType::ElementType &element = BaseType::gridPart().entity(entitySeed);
          const auto geometry = element.geometry();

          const auto &shapeFunctionSet = BaseType::basisSets_.basisFunctionSet(BaseType::agglomeration(), element);

          // compute the boundary terms for the value projection
          for (const auto &intersection : intersections(BaseType::gridPart(), element))
          {
            // ignore edges inside the given polygon
            if (!intersection.boundary() && (BaseType::agglomeration().index(intersection.outside()) == agglomerate))
              continue;
            assert(intersection.conforming());

            const typename BaseType::BasisSetsType::EdgeShapeFunctionSetType edgeShapeFunctionSet
                  = BaseType::basisSets_.edgeBasisFunctionSet(BaseType::agglomeration(), intersection);

            Std::vector<Std::vector<unsigned int>> mask(2,Std::vector<unsigned int>(0)); // contains indices with Phi_mask[i] is attached to given edge
            edgePhiVector[0] = 0;
            edgePhiVector[1] = 0;

            BaseType::interpolation_(intersection, edgeShapeFunctionSet, edgePhiVector, mask);

            auto normal = intersection.centerUnitOuterNormal();

            // now compute int_e Phi^e m_alpha
            typename BaseType::Quadrature1Type quadrature(BaseType::gridPart(), intersection, 2 * polOrder + 1, BaseType::Quadrature1Type::INSIDE);
            for (std::size_t qp = 0; qp < quadrature.nop(); ++qp)
            {
              auto x = quadrature.localPoint(qp);
              auto y = intersection.geometryInInside().global(x);
              const DomainFieldType weight = intersection.geometry().integrationElement(x) * quadrature.weight(qp);
              // need to call shape set scalar each for the correct test functions
              shapeFunctionSet.scalarEach(y, [&](std::size_t alpha, RangeFieldType m)
              {
                if (alpha>=numInnerShapeFunctions)
                {
                  edgeShapeFunctionSet.evaluateEach(x, [&](std::size_t beta,
                        typename BaseType::BasisSetsType::EdgeShapeFunctionSetType::RangeType psi)
                  {
                    for (std::size_t s=0; s<mask[0].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                      // put into correct offset place in constraint RHS matrix
                      RHSconstraintsMatrix[mask[0][s]][alpha] += weight * edgePhiVector[0][beta][s] * psi*normal * m;
                  });
                }
              });
            } // quadrature loop
          } // loop over intersections
        } // loop over triangles in agglomerate
      }
    };

  } // namespace Vem

  namespace Fem
  {
    namespace Capabilities
    {
        template<class GridPart>
        struct hasInterpolation<Vem::CurlFreeVEMSpace<GridPart> > {
            static const bool v = false;
        };
    }
  } // namespace Fem
} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_CURLFREE_HH
