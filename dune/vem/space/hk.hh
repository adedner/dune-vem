#ifndef DUNE_VEM_SPACE_HK_HH
#define DUNE_VEM_SPACE_HK_HH

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

namespace Dune
{
  namespace Vem
  {
    // Internal Forward Declarations
    // -----------------------------

    template<class FunctionSpace, class GridPart,
             bool vectorSpace, bool reduced>
             // bool vectorSpace = false, bool reduced = false>
    class AgglomerationVEMSpace;
    template< class Traits >
    class AgglomerationVEMInterpolation;

    // IsAgglomerationVEMSpace
    // -----------------------

    template<class FunctionSpace, class GridPart, bool vectorSpace, bool reduced>
    struct IsAgglomerationVEMSpace<AgglomerationVEMSpace<FunctionSpace, GridPart, vectorSpace,reduced> >
            : std::integral_constant<bool, true> {
    };

    // AgglomerationVEMSpaceTraits
    // ---------------------------

    template<class FunctionSpace, class GridPart, bool vectorspace, bool reduced>
    struct AgglomerationVEMBasisSets
    {
      typedef GridPart GridPartType;
      static constexpr bool vectorSpace = vectorspace;
      static constexpr int dimDomain = GridPartType::dimension;
      typedef typename GridPart::template Codim<0>::EntityType EntityType;
      typedef typename GridPart::IntersectionType IntersectionType;

      // a scalar function space
      typedef Dune::Fem::FunctionSpace<
              typename FunctionSpace::DomainFieldType, typename FunctionSpace::RangeFieldType,
              dimDomain, 1 > ScalarFunctionSpaceType;

      // scalar BB basis
      typedef Dune::Fem::OrthonormalShapeFunctionSet< ScalarFunctionSpaceType > ONBShapeFunctionSetType;
      typedef BoundingBoxBasisFunctionSet< GridPartType, ONBShapeFunctionSetType > ScalarBBBasisFunctionSetType;

      // vector version of the BB basis for use with vector spaces
      typedef std::conditional_t< vectorSpace,
              Fem::VectorialShapeFunctionSet<ScalarBBBasisFunctionSetType, typename FunctionSpace::RangeType>,
              ScalarBBBasisFunctionSetType
              > BBBasisFunctionSetType;

      // Next we define test function space for the edges
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld-1,1> EdgeFSType;
      typedef Dune::Fem::OrthonormalShapeFunctionSet<EdgeFSType> ScalarEdgeShapeFunctionSetType;

      typedef std::array<std::vector<int>,dimDomain+1> TestSpacesType;

    private:
      // implement three shape functions sets for
      // value: as full basis function set
      // jacobian: with evaluateEach and divergenceEach
      // hessian: with evaluateEach and divergenceEach
      // test: for the inner moments
      // implement edge shape function sets for the testing (value, normal derivative etc)
      struct ShapeFunctionSet
      {
        typedef typename BBBasisFunctionSetType::FunctionSpaceType FunctionSpaceType;
        static const int dimDomain = FunctionSpaceType::DomainType::dimension;
        typedef typename FunctionSpaceType::RangeType RangeType;
        typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;
        typedef typename FunctionSpaceType::HessianRangeType HessianRangeType;
        static const int dimRange = RangeType::dimension;
        static_assert(vectorSpace || dimRange==1);
        ShapeFunctionSet() = default;
        template <class Agglomeration>
        ShapeFunctionSet(bool useOnb, const ONBShapeFunctionSetType& onbSFS,
                         std::size_t numValueSFS, std::size_t numGradSFS, std::size_t numHessSFS,
                         std::size_t innerNumSFS,
                         const Agglomeration &agglomeration, const EntityType &entity)
        : sfs_(entity, agglomeration.index(entity),
               agglomeration.boundingBoxes(), useOnb, onbSFS)
        , numValueShapeFunctions_(numValueSFS)
        , numGradShapeFunctions_(numGradSFS)
        , numHessShapeFunctions_(numHessSFS)
        , numInnerShapeFunctions_(innerNumSFS)
        {}

        int order () const { return sfs_.order();  }

        const BBBasisFunctionSetType &valueBasisSet() const
        {
          return sfs_;
        }

        template< class Point, class Functor >
        void evaluateEach ( const Point &x, Functor functor ) const
        {
          sfs_.evaluateEach(x, functor);
        }
        template< class Point, class Functor >
        void jacobianEach ( const Point &x, Functor functor ) const
        {
          if constexpr (!reduced)
          {
            JacobianRangeType jac(0);
            sfs_.evaluateEach(x, [&](std::size_t alpha, RangeType phi)
            {
              if (alpha<numGradShapeFunctions_)
              {
                for (size_t d=0;d<dimDomain;++d)
                {
                  for (size_t r=0;r<phi.size();++r)
                    jac[r][d] = phi[r];
                  functor(dimDomain*alpha+d,jac);
                  for (size_t r=0;r<phi.size();++r)
                    jac[r][d] = 0;
                }
              }
            });
          }
          else
          {
            sfs_.jacobianEach(x, [&](std::size_t alpha, JacobianRangeType dphi)
            {
              if (alpha>=dimRange) functor(alpha-dimRange,dphi);
            });
          }
        }
        template< class Point, class Functor >
        void hessianEach ( const Point &x, Functor functor ) const
        {
          if constexpr (!reduced)
          {
            HessianRangeType hess(0);
            std::size_t beta = 0;
            sfs_.evaluateEach(x, [&](std::size_t alpha, RangeType phi)
            {
              if (alpha<numHessShapeFunctions_)
              {
                for (size_t d1=0;d1<dimDomain;++d1)
                {
                  for (size_t d2=0;d2<=d1;++d2)
                  {
                    for (size_t r=0;r<phi.size();++r)
                    {
                      hess[r][d1][d2] = phi[r];
                      hess[r][d2][d1] = phi[r];
                    }
                    functor(beta,hess);
                    ++beta;
                    for (size_t r=0;r<phi.size();++r)
                    {
                      hess[r][d1][d2] = 0;
                      hess[r][d2][d1] = 0;
                    }
                  }
                }
              }
            });
          }
          else
          {
            /*
            sfs_.hessianEach(x, [&](std::size_t alpha, HessianRangeType d2phi)
            {
              if (alpha>=(dimDomain+1)*dimRange)
                functor(alpha-(dimDomain+1)*dimRange,d2phi);
            });
            */
          }
        }
        template< class Point, class Functor >
        void evaluateTestEach ( const Point &x, Functor functor ) const
        {
          sfs_.evaluateEach(x, [&](std::size_t alpha, RangeType phi)
          {
            if (alpha < numInnerShapeFunctions_)
              functor(alpha,phi);
          });
        }
        // functor(alpha, psi) with psi in R^r
        //
        // for each g = g_{alpha*dimDomain+s} = m_alpha e_s    (1<=alpha<=numGradSF and 1<=s<=dimDomain)
        // sum_{rj} int_E d_j v_r g_rj = - sum_{rj} int_E v_r d_j g_rj + ...
        //     = - int_E sum_r ( v_r sum_j d_j g_rj )
        //     = - int_E v . psi
        // with psi_r = sum_j d_j g_rj
        //
        // g_{rj} = m_r delta_{js}   (m=m_alpha and fixed s=1,..,dimDomain)
        // psi_r = sum_j d_j g_rj = sum_j d_j m_r delta_{js} = d_s m_r
        template< class Point, class Functor >
        void divJacobianEach( const Point &x, Functor functor ) const
        {
          RangeType divGrad(0);
          if constexpr (!reduced)
          {
            sfs_.jacobianEach(x, [&](std::size_t alpha, JacobianRangeType dphi)
            {
              if (alpha<numGradShapeFunctions_)
                for (size_t s=0;s<dimDomain;++s)
                {
                  for (size_t i=0;i<divGrad.size();++i)
                    divGrad[i] = dphi[i][s];
                  functor(dimDomain*alpha+s, divGrad);
                }
            });
          }
          else
          {
            sfs_.hessianEach(x, [&](std::size_t alpha, HessianRangeType d2phi)
            {
              if (alpha>=dimRange)
              {
                for (size_t i=0;i<divGrad.size();++i)
                {
                  divGrad[i] = 0;
                  for (size_t s=0;s<dimDomain;++s)
                    divGrad[i] += d2phi[i][s][s];
                }
                functor(alpha-dimRange, divGrad);
              }
            });
          }
        }
        // functor(alpha, psi) with psi in R^{r,d}
        //
        // h_{alpha*D^2+d1*D+d2}
        // for each h_r = m_{alpha,r} S_{d1,d2}    (fixed 1<=alpha<=numHessSF and 1<=d1,d2<=dimDomain)
        // sum_{rij} int_E d_ij v_r h_rij = - sum_{rij} int_E d_i v_r d_j h_rij + ...
        //     = - int_E sum_ri (d_i v_r sum_j d_j h_rij )
        //     = - int_E sum_ri d_i v_r psi_ri
        // with psi_ri = sum_j d_j h_rij
        //
        // h_{rij} = m_{alpha,r} (delta_{i,d1}delta_{j,d2}+delta_{j,d1}delta_{i,d2})
        //           (m=m_alpha and fixed ij=1,..,dimDomain)
        // psi_ri = sum_j d_j h_rij
        //        = sum_j d_j m_{alpha,r} (delta_{i,d1}delta_{j,d2}+delta_{j,d1}delta_{i,d2})
        //        = (d_d2 m_{alpha,r} delta_{i,d1} + d_d1 m_{alpha,r} delta_{i,d2}
        template< class Point, class Functor >
        void divHessianEach( const Point &x, Functor functor ) const
        {
          JacobianRangeType divHess(0);
          std::size_t beta=0;
          if constexpr (!reduced)
          {
            sfs_.jacobianEach(x, [&](std::size_t alpha, JacobianRangeType dphi)
            {
              if (alpha<numHessShapeFunctions_)
              {
                for (size_t d1=0;d1<dimDomain;++d1)
                {
                  for (size_t d2=0;d2<=d1;++d2)
                  {
                    divHess = 0;
                    for (size_t r=0;r<dimRange;++r)
                    {
                      divHess[r][d1] = dphi[r][d2];
                      divHess[r][d2] = dphi[r][d1];
                    }
                    functor(beta, divHess);
                    ++beta;
                  }
                }
              }
            });
          }
          else
          {
            // if (sfs_.order()>2) DUNE_THROW( NotImplemented, "hessianEach not implemented for reduced space - needs third order derivative" );
          }
        }

        private:
        BBBasisFunctionSetType sfs_;
        std::size_t numValueShapeFunctions_;
        std::size_t numGradShapeFunctions_;
        std::size_t numHessShapeFunctions_;
        std::size_t numInnerShapeFunctions_;
      };
      struct EdgeShapeFunctionSet
      {
        typedef typename BBBasisFunctionSetType::FunctionSpaceType FunctionSpaceType;
        typedef typename FunctionSpaceType::DomainType DomainType;
        typedef typename FunctionSpaceType::RangeType RangeType;
        typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;
        typedef typename FunctionSpaceType::HessianRangeType HessianRangeType;
        static const int dimDomain = DomainType::dimension;
        static const int dimRange = RangeType::dimension;

        typedef std::conditional_t< vectorSpace,
              Fem::VectorialShapeFunctionSet<ScalarEdgeShapeFunctionSetType, RangeType>,
              ScalarEdgeShapeFunctionSetType > VectorEdgeShapeFunctionSetType;
        typedef typename VectorEdgeShapeFunctionSetType::JacobianRangeType EdgeJacobianRangeType;

        EdgeShapeFunctionSet(const IntersectionType &intersection, const ScalarEdgeShapeFunctionSetType &sfs,
                             unsigned int numEdgeTestFunctions)
        : intersection_(intersection), sfs_(sfs), numEdgeTestFunctions_(numEdgeTestFunctions)
        {}
        template< class Point, class Functor >
        void evaluateEach ( const Point &x, Functor functor ) const
        {
          sfs_.evaluateEach(x,functor);
        }
        template< class Point, class Functor >
        void jacobianEach ( const Point &x, Functor functor ) const
        {
          JacobianRangeType jac;
          const auto &geo = intersection_.geometry();
          const auto &jit = geo.jacobianInverseTransposed(x);
          sfs_.jacobianEach(x, [&](std::size_t alpha, EdgeJacobianRangeType dphi)
          {
            for (std::size_t r=0;r<dimRange;++r)
              jit.mv(dphi[r],jac[r]);
            functor(alpha,jac);
          });
        }
        template< class Point, class Functor >
        void evaluateTestEach ( const Point &x, Functor functor ) const
        {
          sfs_.evaluateEach(x, [&](std::size_t alpha, RangeType phi)
          {
            if (alpha<numEdgeTestFunctions_)
              functor(alpha,phi);
          });
        }
        private:
        const IntersectionType &intersection_;
        VectorEdgeShapeFunctionSetType sfs_;
        unsigned int numEdgeTestFunctions_;
      };

      public:
      typedef ShapeFunctionSet ShapeFunctionSetType;
      typedef EdgeShapeFunctionSet EdgeShapeFunctionSetType;

      AgglomerationVEMBasisSets( const int order,
                                 const TestSpacesType &testSpaces,
                                 int basisChoice )
      // use order2size
      : testSpaces_(testSpaces)
      , useOnb_(basisChoice == 2)
      , dofsPerCodim_(calcDofsPerCodim())
      , onbSFS_(Dune::GeometryType(Dune::GeometryType::cube, dimDomain), order)
      , edgeSFS_( Dune::GeometryType(Dune::GeometryType::cube,dimDomain-1), maxEdgeDegree() )
      , numValueShapeFunctions_( onbSFS_.size()*BBBasisFunctionSetType::RangeType::dimension)
      , numGradShapeFunctions_ (
          !reduced? std::min( numValueShapeFunctions_, sizeONB<0>(std::max(0, order - 1)) )
          : numValueShapeFunctions_-1*BBBasisFunctionSetType::RangeType::dimension
        )
      , numHessShapeFunctions_ (
          !reduced? std::min( numValueShapeFunctions_, sizeONB<0>(std::max(0, order - 2)) )
          : 0 // numValueShapeFunctions_-3*BBBasisFunctionSetType::RangeType::dimension
        )
      , numInnerShapeFunctions_( testSpaces[2][0]<0? 0 : sizeONB<0>(testSpaces[2][0]) )
      , numEdgeTestShapeFunctions_( sizeONB<1>(
                 *std::max_element( testSpaces_[1].begin(), testSpaces_[1].end()) ) )
      {
        auto degrees = edgeDegrees();
        std::cout << "[" << numValueShapeFunctions_ << ","
                  << numGradShapeFunctions_ << ","
                  << numHessShapeFunctions_ << ","
                  << numInnerShapeFunctions_ << "]"
                  << "   edge: ["
                  << edgeSize(0) << "," << edgeSize(1) << ","
                  << numEdgeTestShapeFunctions_ << "]"
                  << " " << degrees[0] << " " << degrees[1]
                  << " max size of edge set: " << edgeSFS_.size()
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
        return ShapeFunctionSet(useOnb_, onbSFS_, numValueShapeFunctions_, numGradShapeFunctions_, numHessShapeFunctions_,
                                numInnerShapeFunctions_,
                                agglomeration,entity);
      }
      template <class Agglomeration>
      EdgeShapeFunctionSetType edgeBasisFunctionSet(
             const Agglomeration &agglomeration, const IntersectionType &intersection) const
      {
        return EdgeShapeFunctionSetType(intersection, edgeSFS_, numEdgeTestShapeFunctions_);
      }
      std::size_t size( std::size_t orderSFS ) const
      {
        if constexpr (!reduced)
        {
          if (orderSFS == 0)
            return numValueShapeFunctions_;
          else if (orderSFS == 1)
            return dimDomain*numGradShapeFunctions_;
          else if (orderSFS == 2)
            return dimDomain*(dimDomain+1)/2*numHessShapeFunctions_;
        }
        else
        {
          if (orderSFS == 0)
            return numValueShapeFunctions_;
          else if (orderSFS == 1)
            return numGradShapeFunctions_;
          else if (orderSFS == 2)
            return numHessShapeFunctions_;
        }
        assert(0);
        return 0;
      }
      int constraintSize() const
      {
        return numInnerShapeFunctions_;
      }
      int vertexSize(int deriv) const
      {
        if (testSpaces_[0][deriv]<0)
          return 0;
        else
          return pow(dimDomain,deriv);
      }
      int innerSize() const
      {
        return numInnerShapeFunctions_;
      }
      int edgeValueMoments() const
      {
        // returns order of edge moments up to P_k where k is the entry in dof tuple
        return testSpaces_[1][0];
      }
      std::size_t edgeSize(int deriv) const
      {
        auto degrees = edgeDegrees();
        return degrees[deriv] < 0 ? 0 : sizeONB<1>( degrees[deriv] );
        /* Dune::Fem::OrthonormalShapeFunctions<1>::size( degrees[deriv] )
           * BBBasisFunctionSetType::RangeType::dimension; */
      }

      ////////////////////////////
      // used in interpolation
      ////////////////////////////
      std::size_t edgeSize() const
      {
        return edgeSFS_.size() * BBBasisFunctionSetType::RangeType::dimension;
      }
      std::size_t numEdgeTestShapeFunctions() const
      {
        return numEdgeTestShapeFunctions_;
      }
      const TestSpacesType &testSpaces() const
      {
        return testSpaces_;
      }
      template <int dim>
      std::size_t order2size(unsigned int deriv) const
      {
        if (testSpaces_[dim].size()<=deriv || testSpaces_[dim][deriv]<0)
          return 0;
        else
        {
          if constexpr (dim>0)
            return Dune::Fem::OrthonormalShapeFunctions<dim>::
              size(testSpaces_[dim][deriv]);
          else
            return pow(dimDomain,deriv);
        }
      }
      int vectorDofs(int dim) const
      {
        // return 1 for scalar dofs or dimRange for vector dofs
        // want: (basisFunctionSet::RangeType::dimension == RangeType::dimension) ? RangeType::dimension : 1
        return vectorSpace ? BBBasisFunctionSetType::FunctionSpaceType::RangeType::dimension : 1;
      }

      private:
      Std::vector<int> edgeDegrees() const
      {
        assert( testSpaces_[2].size()<2 );
        Std::vector<int> degrees(2, -1);
        for (std::size_t i=0;i<testSpaces_[0].size();++i)
          degrees[i] += 2*(testSpaces_[0][i]+1);
        if (testSpaces_[0].size()>1 && testSpaces_[0][1]>-1) // add tangential derivatives
          degrees[0] += 2;
        for (std::size_t i=0;i<testSpaces_[1].size();++i)
          degrees[i] += std::max(0,testSpaces_[1][i]+1);
        return degrees;
      }
      std::size_t maxEdgeDegree() const
      {
        auto degrees = edgeDegrees();
        return *std::max_element(degrees.begin(),degrees.end());
      }

      template <int codim>
      static std::size_t sizeONB(std::size_t order)
      {
        return Dune::Fem::OrthonormalShapeFunctions<dimDomain - codim> :: size(order) *
               BBBasisFunctionSetType::RangeType::dimension;
      }

      std::array< std::pair< int, unsigned int >, dimDomain+1 > calcDofsPerCodim () const
      {
        int vSize = 0;
        int eSize = 0;
        int iSize = 0;
        for (size_t i=0;i<testSpaces_[0].size();++i)
          vSize += order2size<0>(i);
        for (size_t i=0;i<testSpaces_[1].size();++i)
          eSize += order2size<1>(i);
        for (size_t i=0;i<testSpaces_[2].size();++i)
          iSize += order2size<2>(i);
        return std::array< std::pair< int, unsigned int >, dimDomain+1 >
               { std::make_pair( dimDomain,   vSize ),
                 std::make_pair( dimDomain-1, eSize ),
                 std::make_pair( dimDomain-2, iSize ) };
      }

      // note: the actual shape function set depends on the entity so
      // we can only construct the underlying monomial basis in the ctor
      const TestSpacesType testSpaces_;
      const bool useOnb_;
      std::array< std::pair< int, unsigned int >, dimDomain+1 > dofsPerCodim_;
      const ONBShapeFunctionSetType onbSFS_;
      const ScalarEdgeShapeFunctionSetType edgeSFS_;
      const std::size_t numValueShapeFunctions_;
      const std::size_t numGradShapeFunctions_;
      const std::size_t numHessShapeFunctions_;
      const std::size_t numInnerShapeFunctions_;
      const std::size_t numEdgeTestShapeFunctions_;
    };



    template<class FunctionSpace, class GridPart,
             bool vectorspace, bool reduced>
    struct AgglomerationVEMSpaceTraits
    {
      typedef AgglomerationVEMBasisSets<FunctionSpace,GridPart,vectorspace,reduced> BasisSetsType;

      static const bool vectorSpace = vectorspace;
      friend class AgglomerationVEMSpace<FunctionSpace, GridPart, vectorSpace, reduced>;

      typedef AgglomerationVEMSpace<FunctionSpace, GridPart, vectorSpace, reduced> DiscreteFunctionSpaceType;

      typedef GridPart GridPartType;

      static const int dimension = GridPartType::dimension;
      static const int codimension = 0;
      static const int dimDomain = FunctionSpace::DomainType::dimension;
      static const int dimRange = FunctionSpace::RangeType::dimension;
      // static const int baseRangeDimension = vectorSpace ? dimRange : 1;

      typedef typename GridPartType::template Codim<codimension>::EntityType EntityType;
      typedef FunctionSpace FunctionSpaceType;

      // vem basis function sets
      typedef VEMBasisFunctionSet <EntityType, typename BasisSetsType::ShapeFunctionSetType> ScalarBasisFunctionSetType;
      typedef std::conditional_t< vectorSpace,
              ScalarBasisFunctionSetType,
              Fem::VectorialBasisFunctionSet<ScalarBasisFunctionSetType, typename FunctionSpaceType::RangeType>
              > BasisFunctionSetType;

      // types for the mapper
      typedef Hybrid::IndexRange<int, dimRange> LocalBlockIndices;
      typedef VemAgglomerationIndexSet <GridPartType> IndexSetType;
      typedef AgglomerationDofMapper <GridPartType, IndexSetType> BlockMapperType;

      // static const int baseBlockSize = (BasisFunctionSetType::RangeType::dimension == RangeType::dimension) ? LocalBlockIndices::size() : 1;
      static const int baseBlockSize = vectorSpace ? LocalBlockIndices::size() : 1;

      template<class DiscreteFunction, class Operation = Fem::DFCommunicationOperation::Copy>
      struct CommDataHandle {
          typedef Operation OperationType;
          typedef Fem::DefaultCommunicationHandler <DiscreteFunction, Operation> Type;
      };

      template <class T>
      using InterpolationType = AgglomerationVEMInterpolation<T>;
    };

    // AgglomerationVEMSpace
    // ---------------------
    template<class FunctionSpace, class GridPart,
             bool vectorSpace, bool reduced>
    struct AgglomerationVEMSpace
    : public DefaultAgglomerationVEMSpace< AgglomerationVEMSpaceTraits<FunctionSpace,GridPart,vectorSpace,reduced> >
    {
      typedef AgglomerationVEMSpaceTraits<FunctionSpace,GridPart,vectorSpace,reduced> TraitsType;
      typedef DefaultAgglomerationVEMSpace<TraitsType> BaseType;
      typedef Agglomeration<GridPart> AgglomerationType;
      typedef typename TraitsType::FunctionSpaceType FunctionSpaceType;
      typedef typename FunctionSpaceType::DomainFieldType DomainFieldType;
      AgglomerationVEMSpace(AgglomerationType &agglomeration,
          const unsigned int polOrder,
          const typename TraitsType::BasisSetsType::TestSpacesType &testSpaces,
          int basisChoice,
          bool edgeInterpolation)
      : BaseType(agglomeration,polOrder,
                 typename TraitsType::BasisSetsType(polOrder, testSpaces, basisChoice),
                 basisChoice,edgeInterpolation)
      {
        if (basisChoice != 3) // !!!!! get order information from BasisSets
          BaseType::agglomeration().onbBasis(polOrder);
        BaseType::update(true);
      }

    protected:
      virtual void setupConstraintRHS(const Std::vector<Std::vector<typename BaseType::ElementSeedType> > &entitySeeds, unsigned int agglomerate,
                                    Dune::DynamicMatrix<DomainFieldType> &RHSconstraintsMatrix, double volume) override
      {
        //////////////////////////////////////////////////////////////////////////
        /// Fix RHS constraints for value projection /////////////////////////////
        //////////////////////////////////////////////////////////////////////////

        static constexpr int dimRange = TraitsType::dimRange;
        static constexpr int blockSize = TraitsType::vectorSpace ? dimRange : 1;
        const std::size_t numShapeFunctions = BaseType::basisSets_.size(0);
        const std::size_t numDofs = BaseType::blockMapper().numDofs(agglomerate) * blockSize;
        const std::size_t numConstraintShapeFunctions = BaseType::basisSets_.constraintSize();
        const std::size_t numInnerShapeFunctions = BaseType::basisSets_.innerSize();

        assert( numDofs == RHSconstraintsMatrix.rows() );
        assert( numInnerShapeFunctions == RHSconstraintsMatrix.cols() );
        assert( numInnerShapeFunctions <= numConstraintShapeFunctions );
        if (numConstraintShapeFunctions == 0) return;

        // first fill in entries relating to inner dofs (alpha < inner shape functions)
        for ( int beta=0; beta<numDofs; ++beta)
        {
          // TODO
          // don't need loop use
          // int alpha = beta - numDofs + numInnerShapeFunctions;
          // if (alpha>=0) RHSconstraintsMatrix[ beta ][ alpha ] = volume;
          // possibly even fix loop for beta
          for (int alpha=0; alpha<numInnerShapeFunctions; ++alpha)
          {
            if( beta - numDofs + numInnerShapeFunctions == alpha )
              RHSconstraintsMatrix[ beta ][ alpha ] = volume;
          }
        }
      }
    };

    //////////////////////////////////////////////////////////////////////////////
    // Interpolation classes
    //////////////////////////////////////////////////////////////////////////////

    // AgglomerationVEMInterpolation
    // -----------------------------
    /* Methods:
      // interpolation of a local function
      template< class LocalFunction, class LocalDofVector >
      void operator() ( const ElementType &element, const LocalFunction &localFunction,
                        LocalDofVector &localDofVector ) const
      // set mask for active (and type of) dofs - needed for DirichletConstraints
      // Note: this is based on the block dof mapper approach, i.e., one
      //       entry in the mask per block
      void operator() ( const ElementType &element, Std::vector<char> &mask) const

      // apply all dofs to a basis function set (A=L(B))
      template< class BasisFunctionSet, class LocalDofMatrix >
      void interpolateBasis ( const ElementType &element,
                   const BasisFunctionSet &basisFunctionSet, LocalDofMatrix &localDofMatrix ) const
      // setup constraints rhs for value projection CLS problem
      template <class DomainFieldType>
      void valueL2constraints(unsigned int beta, double volume,
                              Dune::DynamicMatrix<DomainFieldType> &D,
                              Dune::DynamicVector<DomainFieldType> &d)
      // interpolate given shape function set on intersection (needed for gradient projection)
      // Note: this fills in the full mask and localDofs, i.e., not only for each block
      template< class EdgeShapeFunctionSet >
      void operator() (const IntersectionType &intersection,
                       const EdgeShapeFunctionSet &edgeShapeFunctionSet, Std::vector < Dune::DynamicMatrix<double> > &localDofVectorMatrix,
                       Std::vector<Std::vector<unsigned int>> &mask) const
      // size of the edgePhiVector
      std::size_t edgeSize(int deriv) const
    */

    template< class Traits >
    class AgglomerationVEMInterpolation
    {
      typedef AgglomerationVEMInterpolation< Traits > ThisType;

    public:
      typedef typename Traits::BasisSetsType BasisSetsType;
      typedef typename BasisSetsType::ShapeFunctionSetType::FunctionSpaceType FunctionSpaceType;
      typedef typename FunctionSpaceType::RangeType RangeType;
      typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;
      typedef typename FunctionSpaceType::HessianRangeType HessianRangeType;
      typedef typename Traits::IndexSetType IndexSetType;
      typedef typename IndexSetType::ElementType ElementType;
      typedef typename IndexSetType::GridPartType GridPartType;
      typedef typename GridPartType::IntersectionType IntersectionType;

      static constexpr int blockSize = Traits::vectorSpace?
                                       Traits::LocalBlockIndices::size() : 1;
      static const int dimension = IndexSetType::dimension;
      static const int baseBlockSize = Traits::baseBlockSize;
    private:
      typedef Dune::Fem::ElementQuadrature<GridPartType,0> InnerQuadratureType;
      typedef Dune::Fem::ElementQuadrature<GridPartType,1> EdgeQuadratureType;

      typedef typename ElementType::Geometry::ctype ctype;

    public:
      explicit AgglomerationVEMInterpolation ( const IndexSetType &indexSet,
                                               const BasisSetsType &basisSets,
                                               unsigned int polOrder, bool useOnb ) noexcept
        : indexSet_( indexSet )
        , basisSets_( basisSets )
        , polOrder_( polOrder )
        , useOnb_(useOnb)
      {}

      const GridPartType &gridPart() const { return indexSet_.agglomeration().gridPart(); }

      template< class LocalFunction, class LocalDofVector >
      void operator() ( const ElementType &element, const LocalFunction &localFunction,
                        LocalDofVector &localDofVector ) const
      {
        // the interpolate__ method handles the 'vector valued' case
        // calling the interpolate_ method for each component - the actual
        // work is done in the interpolate_ method
        interpolate__(element,localFunction,localDofVector, Dune::PriorityTag<LocalFunction::RangeType::dimension>() );
      }

      // fill a mask vector providing the information which dofs are
      // 'active' on the given element, i.e., are attached to a given
      // subentity of this element. Needed for dirichlet boundary data for
      // example
      // Note: this returns the same mask independent of the baseRangeDimension,
      //       i.e., vector extension has to be done on the calling side
      void operator() ( const ElementType &element, Std::vector<char> &mask) const
      {
        std::fill(mask.begin(),mask.end(),-1);
        auto vertex = [&] (int poly,auto i,int k,int numDofs)
        {
          k /= baseBlockSize;
          // mask[k] = 1;
          // ++k;
          // basisSets_.BBBasisFunctionSetType::RangeType::dimension
          for (int r=0; r<2; ++r)
            mask[k+r] = 1;
          k += 2;

          if (order2size<0>(1)>0)
          {
              mask[k]   = 2;
              mask[k+1] = 2;
          }
        };
        auto edge = [&] (int poly,auto i,int k,int numDofs)
        {
          k /= baseBlockSize;
#ifndef NDEBUG
          auto kStart = k;
#endif
          for (std::size_t alpha=0;alpha<basisSets_.numEdgeTestShapeFunctions()/baseBlockSize;++alpha)
          {
            // if (alpha < basisSets_.template order2size<1>(0)*2)
            if (alpha < basisSets_.template order2size<1>(0)*2)
            {
              mask[k] = 1;
              ++k;
            }
            if (alpha < basisSets_.template order2size<1>(1))
            {
              mask[k] = 2;
              ++k;
            }
          }
          // assert(k-kStart == numDofs/baseRangeDimension);
          // assert( numDofs == basisSets_.template order2size<1>(0) );
          // std::fill(mask.begin()+k,mask.begin()+k+numDofs,1);
        };
        auto inner = [&mask] (int poly,auto i,int k,int numDofs)
        {
          // ???? assert( basisSets_.innerTestSize() == numDofs );
          k /= baseBlockSize;
          std::fill(mask.begin()+k,mask.begin()+k+numDofs/baseBlockSize,1);
        };
        apply(element,vertex,edge,inner);
        // assert( std::none_of(mask.begin(),mask.end(), [](char m){return // m==-1;}) ); // ???? needs investigation - issue with DirichletBCs
      }

      // preform interpolation of a full shape function set filling a transformation matrix
      template< class BasisFunctionSet, class LocalDofMatrix >
      void interpolateBasis ( const ElementType &element,
                   const BasisFunctionSet &basisFunctionSet, LocalDofMatrix &localDofMatrix ) const
      {
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );

        // use the bb set for this polygon for the inner testing space
        const auto &innerShapeFunctionSet = basisSets_.basisFunctionSet( indexSet_.agglomeration(), element );

        // define the corresponding vertex,edge, and inner parts of the interpolation
        auto vertex = [&] (int poly,int i,int k,int numDofs)
        { //!TS add derivatives at vertex for conforming space
          const auto &x = refElement.position( i, dimension );
          basisFunctionSet.evaluateEach( x, [ &localDofMatrix, k ] ( std::size_t alpha, typename BasisFunctionSet::RangeType phi ) {
              if (alpha < localDofMatrix[k].size())
                for (int r=0;r<phi.dimension;++r)
                  localDofMatrix[ k+r ][ alpha ] = phi[ r ];
            } );
          k += BasisFunctionSet::RangeType::dimension;
          if (order2size<0>(1)>0)
            basisFunctionSet.jacobianEach( x, [ & ] ( std::size_t alpha, typename BasisFunctionSet::JacobianRangeType dphi ) {
              assert( dphi[0].dimension == 2 );
              if (alpha < localDofMatrix[k+1].size())
              {
                for (int r=0;r<dphi.rows;++r)
                {
                  localDofMatrix[ k+2*r ][ alpha ]   = dphi[r][ 0 ] * indexSet_.vertexDiameter(element, i);
                  localDofMatrix[ k+2*r+1 ][ alpha ] = dphi[r][ 1 ] * indexSet_.vertexDiameter(element, i);
                }
              }
            } );
        };
        auto edge = [&,this] (int poly,auto intersection,int k,int numDofs)
        { //!TS add nomral derivatives
          int kStart = k;
          const auto &edgeBFS = basisSets_.edgeBasisFunctionSet( indexSet_.agglomeration(), intersection );
          // int edgeNumber = intersection.indexInInside();
          EdgeQuadratureType edgeQuad( gridPart(), intersection, 2*polOrder_, EdgeQuadratureType::INSIDE );
          auto normal = intersection.centerUnitOuterNormal();
          if (intersection.neighbor()) // we need to check the orientation of the normal
            if (indexSet_.index(intersection.inside()) > indexSet_.index(intersection.outside()))
              normal *= -1;
          for (unsigned int qp=0;qp<edgeQuad.nop();++qp)
          {
            k = kStart;
            auto x = edgeQuad.localPoint(qp);
            auto y = intersection.geometryInInside().global(x);
            double weight = edgeQuad.weight(qp) *
                            intersection.geometry().integrationElement(x);
            edgeBFS.evaluateTestEach(x,
                [&](std::size_t alpha, RangeType phi ) {
                if (alpha < order2size<1>(0))
                {
                  basisFunctionSet.evaluateEach( y,
                    [ & ] ( std::size_t beta, typename BasisFunctionSet::RangeType value )
                    {
                      assert(k<localDofMatrix.size());
                      assert(beta<localDofMatrix[k].size());
                      localDofMatrix[ k ][ beta ] += value*phi * weight /
                                                     intersection.geometry().volume();

                    }
                  );
                  ++k;
                }
                if (alpha < order2size<1>(1))
                {
                  basisFunctionSet.jacobianEach( y,
                    [ & ] ( std::size_t beta, typename BasisFunctionSet::JacobianRangeType dvalue )
                    {
                      // we assume here that jacobianEach is in global
                      // space so the jit is not applied
                      assert(k<localDofMatrix.size());
                      RangeType dn;
                      dvalue.mv(normal, dn);
                      assert( dn[0] == dvalue[0]*normal );

                      localDofMatrix[ k ][ beta ] += dn*phi * weight;
                    }
                  );
                  ++k;
                }
              }
            );
          }
        };
        auto inner = [&] (int poly,int i,int k,int numDofs)
        {
          // ???? assert(numDofs == basisSets_.innerSize());
          InnerQuadratureType innerQuad( element, 2*polOrder_ );
          for (int qp=0;qp<innerQuad.nop();++qp)
          {
            auto y = innerQuad.point(qp);
            double weight = innerQuad.weight(qp) *
                            element.geometry().integrationElement(y) /
                            indexSet_.volume(poly);
            basisFunctionSet.evaluateEach( innerQuad[qp],
              [ & ] ( std::size_t beta, typename BasisFunctionSet::RangeType value )
              {
                innerShapeFunctionSet.evaluateTestEach( innerQuad[qp],
                  [&](std::size_t alpha, RangeType phi ) {
                    int kk = alpha+k;
                    assert(kk<localDofMatrix.size());
                    localDofMatrix[ kk ][ beta ] += value*phi * weight;
                  }
                );
              }
            );
          }
        };
        apply(element,vertex,edge,inner);
      }

      // interpolate the full shape function set on intersection needed for
      // the gradient projection matrix
      // Note: for a vector valued space this fills in the full 'baseRangeDimension' mask and localDofs
      template< class EdgeShapeFunctionSet >
      void operator() (const IntersectionType &intersection,
                       const EdgeShapeFunctionSet &edgeShapeFunctionSet, Std::vector < Dune::DynamicMatrix<double> > &localDofVectorMatrix,
                       Std::vector<Std::vector<unsigned int>> &mask) const
      {
        for (std::size_t i=0;i<mask.size();++i)
          mask[i].clear();
        const ElementType &element = intersection.inside();
        const auto &edgeBFS = basisSets_.edgeBasisFunctionSet( indexSet_.agglomeration(), intersection );
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );
        int edgeNumber = intersection.indexInInside();
        const auto &edgeGeo = refElement.template geometry<1>(edgeNumber);
        /**/ // Question: is it correct that the nomral and derivatives are not needed here
        auto normal = intersection.centerUnitOuterNormal();
        double flipNormal = 1.;
        if (intersection.neighbor()) // we need to check the orientation of the normal
          if (indexSet_.index(intersection.inside()) > indexSet_.index(intersection.outside()))
          {
            normal *= -1;
            flipNormal = -1;
          }

        /**/
        Std::vector<std::size_t> entry(localDofVectorMatrix.size(), 0);

        // define the three relevant part of the interpolation, i.e.,
        // vertices,edges - no inner needed since only doing interpolation
        // on intersectionn
        auto vertex = [&] (int poly,int i,int k,int numDofs)
        { //!TS add derivatives at vertex (probably only normal component - is the mask then correct?)
          // std::cout << "vertex:" << poly << "," << i << "," << k << "," << numDofs
          //           << " | " << entry[0] << std::endl;
          const auto &x = edgeGeo.local( refElement.position( i, dimension ) );
          edgeShapeFunctionSet.evaluateEach( x, [ &localDofVectorMatrix, &entry ] ( std::size_t alpha, typename EdgeShapeFunctionSet::RangeType phi ) {
              assert( entry[0] < localDofVectorMatrix[0].size() );
              if (alpha < localDofVectorMatrix[0][ entry[0] ].size())
                for (int r=0;r<phi.dimension;++r)
                  localDofVectorMatrix[0][ entry[0]+r ][ alpha ] = phi[ r ];
            } );
          entry[0] += EdgeShapeFunctionSet::RangeType::dimension;
          if (order2size<0>(1)>0)
          {
            edgeShapeFunctionSet.jacobianEach( x, [ & ] ( std::size_t alpha, typename EdgeShapeFunctionSet::JacobianRangeType dphi ) {
              assert( entry[0] < localDofVectorMatrix[0].size() );
              assert( dphi[0].dimension == 1 );
              // note: edge sfs in reference coordinate so apply scaling 1/|S|
              if (alpha < localDofVectorMatrix[0][entry[0]].size())
                for (int r=0;r<dphi.rows;++r)
                  localDofVectorMatrix[ 0 ][ entry[0]+r ][ alpha ] = dphi[r][0] / intersection.geometry().volume()
                                                                   * indexSet_.vertexDiameter(element, i);
            } );
            edgeShapeFunctionSet.evaluateEach( x, [ & ] ( std::size_t alpha, typename EdgeShapeFunctionSet::RangeType phi ) {
              assert( entry[1] < localDofVectorMatrix[1].size() );
              if (alpha < localDofVectorMatrix[1][entry[1]].size())
                for (int r=0;r<phi.dimension;++r)
                  localDofVectorMatrix[ 1 ][ entry[1]+r ][ alpha ] = phi[r]*flipNormal
                                                                 * indexSet_.vertexDiameter(element, i);
            } );
            entry[0] += EdgeShapeFunctionSet::RangeType::dimension;
            entry[1] += EdgeShapeFunctionSet::RangeType::dimension;
          }
        };
        auto edge = [&] (int poly,auto intersection,int k,int numDofs)
        { //!TS add normal derivatives
          // std::cout << "edge:" << poly << "," << k << "," << numDofs
          //           << " | " << entry[0] << std::endl;
          EdgeQuadratureType edgeQuad( gridPart(),
                intersection, 2*polOrder_, EdgeQuadratureType::INSIDE );
          for (unsigned int qp=0;qp<edgeQuad.nop();++qp)
          {
            auto x = edgeQuad.localPoint(qp);
            auto xx = x;
            double weight = edgeQuad.weight(qp) *
                            intersection.geometry().integrationElement(x);
            edgeShapeFunctionSet.evaluateEach( x, [ & ] ( std::size_t beta, typename EdgeShapeFunctionSet::RangeType value ) {
                edgeBFS.evaluateTestEach( xx,
                  [&](std::size_t alpha, typename EdgeShapeFunctionSet::RangeType phi ) {
                    /* std::cout << "alpha=" << alpha << " row=" << entry[0]+alpha << " | "
                              << value << "*" << phi << "=" << value*phi
                              << std::endl; */
                    //!TS add alpha<...
                    if (alpha < order2size<1>(0) && beta < edgeSize(0))
                    {
                      assert( entry[0]+alpha < localDofVectorMatrix[0].size() );
                      localDofVectorMatrix[0][ entry[0]+alpha ][ beta ] += value*phi * weight /
                                               intersection.geometry().volume();
                    }
                    // FIX ME
                    if (alpha < order2size<1>(1) && beta < edgeSize(1))
                    {
                      assert( entry[1]+alpha < localDofVectorMatrix[1].size() );
                      localDofVectorMatrix[1][ entry[1]+alpha ][ beta ] += value*phi * weight * flipNormal;
                    }
                  }
                );
              }
            );
          }
          entry[0] += order2size<1>(0);
          entry[1] += order2size<1>(1);
        };

        applyOnIntersection(intersection,vertex,edge,mask);

        // std::cout << "final entry=" << entry[0] << std::endl;

        assert( entry[0] == localDofVectorMatrix[0].size() );
        assert( entry[1] == localDofVectorMatrix[1].size() );

        //////////////////////////////////////////////////////////////////////////////////
        auto tau = intersection.geometry().corner(1);
        tau -= intersection.geometry().corner(0);
        if (localDofVectorMatrix[0].size() > 0)
        {
          try
          {
            localDofVectorMatrix[0].invert();
          }
          catch (const FMatrixError&)
          {
            std::cout << "localDofVectorMatrix.invert() failed!\n";
            for (std::size_t alpha=0;alpha<localDofVectorMatrix[0].size();++alpha)
            {
              for (std::size_t beta=0;beta<localDofVectorMatrix[0][alpha].size();++beta)
                std::cout << localDofVectorMatrix[0][alpha][beta] << " ";
              std::cout << std::endl;
            }
            assert(0);
            throw FMatrixError();
          }

          if (mask[1].size() > edgeSize(1))
          { // need to take tangential derivatives at vertices into account
            assert(mask[0].size() == edgeSize(0)+2);
            auto A = localDofVectorMatrix[0];
            localDofVectorMatrix[0].resize(edgeSize(0), mask[0].size(), 0);
            // vertex basis functions (values)
            for (std::size_t j=0;j<edgeSize(0);++j)
            {
              localDofVectorMatrix[0][j][0] = A[j][0];
              localDofVectorMatrix[0][j][3] = A[j][2];
            }
            // vertex basis functions (tangential derivatives)
            // TODO: add baseRangeDimension
            for (std::size_t j=0;j<edgeSize(0);++j)
            {
              localDofVectorMatrix[0][j][1] = A[j][1]*tau[0];
              localDofVectorMatrix[0][j][2] = A[j][1]*tau[1];
              localDofVectorMatrix[0][j][4] = A[j][3]*tau[0];
              localDofVectorMatrix[0][j][5] = A[j][3]*tau[1];
            }
            for (std::size_t i=6;i<mask[0].size();++i)
              for (std::size_t j=0;j<edgeSize(0);++j)
              {
                assert( i-2 < A[j].size() );
                localDofVectorMatrix[0][j][i] = A[j][i-2];
              }
          }
        }
        if (localDofVectorMatrix[1].size() > 0)
        {
          localDofVectorMatrix[1].invert();
          if (mask[1].size() > edgeSize(1))
          {
            assert(mask[1].size() == edgeSize(1)+2);
            auto A = localDofVectorMatrix[1];
            localDofVectorMatrix[1].resize(edgeSize(1), mask[1].size(), 0);
            std::size_t i=0;
            // vertex basis functions
            for (;i<4;i+=2)
            {
              for (std::size_t j=0;j<edgeSize(1);++j)
              {
                localDofVectorMatrix[1][j][i]   = A[j][i/2]*normal[0];
                localDofVectorMatrix[1][j][i+1] = A[j][i/2]*normal[1];
              }
            }
            for (;i<mask[1].size();++i)
              for (std::size_t j=0;j<edgeSize(1);++j)
                localDofVectorMatrix[1][j][i] = A[j][i-2];
          }
        }
        /* ////////////////////////////////////////////////////////////////////// */
        // It might be necessary to flip the vertex entries in the masks around
        // due to a twist in the intersection.
        // At the moment this is done by checking that the
        /* ////////////////////////////////////////////////////////////////////// */
        {
          auto otherTau = element.geometry().corner(
                       refElement.subEntity(intersection.indexInInside(),1,1,2)
                     );
          otherTau -= element.geometry().corner(
                        refElement.subEntity(intersection.indexInInside(),1,0,2)
                      );
          otherTau /= otherTau.two_norm();
          if (basisSets_.vertexSize(0) > 0) // vertices might have to be flipped
          {
            if (otherTau*tau<0)
            {
              if (mask[1].size() > edgeSize(1))
              {
                // swap vertex values and tangential derivatives
                for (int r=0;r<3*basisSets_.vectorDofs(0);++r)
                  std::swap(mask[0][r], mask[0][3*basisSets_.vectorDofs(0)+r]);
                // swap normal derivatives at vertices
                for (int r=0;r<2*basisSets_.vectorDofs(0);++r)
                  std::swap(mask[1][r], mask[1][2*basisSets_.vectorDofs(0)+r]);
              }
              else
              {
                // swap vertex values
                for (int r=0;r<EdgeShapeFunctionSet::RangeType::dimension;++r)
                  std::swap(mask[0][r], mask[0][basisSets_.vectorDofs(0)+r]);
              }
            }
          }
        }
      }

      std::size_t edgeSize(int deriv) const
      {
        return basisSets_.edgeSize(deriv);
      }

    private:
      template <int dim>
      std::size_t order2size(unsigned int deriv) const
      {
        return basisSets_.template order2size<dim>(deriv) * basisSets_.vectorDofs(dim);
      }

      void getSizesAndOffsets(int poly,
                  int &vertexSize,
                  int &edgeOffset, int &edgeSize,
                  int &innerOffset, int &innerSize) const
      {
        auto dofs   = basisSets_.dofsPerCodim();  // assume always three entries in dim order (i.e. 2d)
        assert(dofs.size()==3);
        vertexSize  = dofs[0].second*blockSize;
        edgeOffset  = indexSet_.subAgglomerates(poly,dimension)*vertexSize;
        edgeSize    = dofs[1].second*blockSize;
        innerOffset = edgeOffset + indexSet_.subAgglomerates(poly,dimension-1)*edgeSize;
        innerSize   = dofs[2].second*blockSize;
      }

      // carry out actual interpolation giving the three components, i.e.,
      // for the vertex, edge, and inner parts.
      // This calls these interpolation operators with the correct indices
      // to fill the dof vector or the matrix components
      template< class Vertex, class Edge, class Inner>
      void apply ( const ElementType &element,
          const Vertex &vertex, const Edge &edge, const Inner &inner) const
      {
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );
        const int poly = indexSet_.index( element );
        int vertexSize, edgeOffset,edgeSize, innerOffset,innerSize;
        getSizesAndOffsets(poly, vertexSize,edgeOffset,edgeSize,innerOffset,innerSize);

        // vertex dofs
        //!TS needs changing
        if (basisSets_.vertexSize(0) > 0)
        {
          for( int i = 0; i < refElement.size( dimension ); ++i )
          {
            const int k = indexSet_.localIndex( element, i, dimension) * vertexSize;
            if ( k >= 0 ) // is a 'real' vertex of the polygon
              vertex(poly,i,k,1);
          }
        }
        //!TS needs changing
        if (order2size<1>(0)>0 ||
            order2size<1>(1)>0)
        {
          // to avoid any issue with twists we use an intersection iterator
          // here instead of going over the edges
          auto it = gridPart().ibegin( element );
          const auto endit = gridPart().iend( element );
          for( ; it != endit; ++it )
          {
            const auto& intersection = *it;
            const int i = intersection.indexInInside();
            const int k = indexSet_.localIndex( element, i, dimension-1 )*edgeSize + edgeOffset; //
            if ( k>=edgeOffset ) // 'real' edge of polygon
              edge(poly,intersection,k,edgeSize);
          }
        }
        //! needs changing
        if (basisSets_.innerSize() > 0)
        {
          // inner dofs
          const int k = indexSet_.localIndex( element, 0, 0 )*innerSize + innerOffset;
          inner(poly,0,k,innerSize);
        }
      }

      // perform interpolation for a single localized function, calls the
      // 'apply' method with the correct 'vertex','edge','inner' functions
      template< class LocalFunction, class LocalDofVector >
      void interpolate_ ( const ElementType &element, const LocalFunction &localFunction,
                          LocalDofVector &localDofVector ) const
      {
        typename LocalFunction::RangeType value;
        typename LocalFunction::JacobianRangeType dvalue;
        // assert( value.dimension == baseRangeDimension );
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );

        // use the bb set for this polygon for the inner testing space
        const auto &innerShapeFunctionSet = basisSets_.basisFunctionSet( indexSet_.agglomeration(), element );

        // define the vertex,edge, and inner parts of the interpolation
        auto vertex = [&] (int poly,auto i,int k,int numDofs)
        { //!TS vertex derivatives
          const auto &x = refElement.position( i, dimension );
          localFunction.evaluate( x, value );
          //! SubDofWrapper does not have size assert( k < localDofVector.size() );
          for (int r=0;r<value.dimension;++r)
            localDofVector[ k+r ] = value[ r ];
          k += value.dimension;
          if (order2size<0>(1)>0)
          {
            localFunction.jacobian( x, dvalue );
            for (int r=0;r<value.dimension;++r)
            {
              localDofVector[ k+2*r ]   = dvalue[ r ][ 0 ] * indexSet_.vertexDiameter(element, i);
              localDofVector[ k+2*r+1 ] = dvalue[ r ][ 1 ] * indexSet_.vertexDiameter(element, i);
            }
          }
        };
        auto edge = [&] (int poly,auto intersection,int k,int numDofs)
        { //!TS edge derivatives
          int kStart = k;
          const auto &edgeBFS = basisSets_.edgeBasisFunctionSet( indexSet_.agglomeration(), intersection );
          auto normal = intersection.centerUnitOuterNormal();
          if (intersection.neighbor()) // we need to check the orientation of the normal
            if (indexSet_.index(intersection.inside()) > indexSet_.index(intersection.outside()))
              normal *= -1;
          EdgeQuadratureType edgeQuad( gridPart(),
                intersection, 2*polOrder_, EdgeQuadratureType::INSIDE );
          for (unsigned int qp=0;qp<edgeQuad.nop();++qp)
          {
            k = kStart;
            auto x = edgeQuad.localPoint(qp);
            auto y = intersection.geometryInInside().global(x);
            localFunction.evaluate( y, value );
            double weight = edgeQuad.weight(qp) * intersection.geometry().integrationElement(x);
            if (order2size<1>(1)>0)
              localFunction.jacobian( y, dvalue);
            typename LocalFunction::RangeType dnvalue;
            dvalue.mv(normal,dnvalue);
            assert( dnvalue[0] == dvalue[0]*normal );
            edgeBFS.evaluateTestEach(x,
              [&](std::size_t alpha, typename LocalFunction::RangeType phi ) {
                //! SubDofWrapper has no size assert( kk < localDofVector.size() );
                if (alpha < order2size<1>(0))
                {
                  //! see above assert(k < localDofVector.size() );
                  localDofVector[ k ] += value*phi * weight
                                         / intersection.geometry().volume();
                  ++k;
                }
                if (alpha < order2size<1>(1))
                {
                  //! see above assert(k < localDofVector.size() );
                  localDofVector[ k ] += dnvalue*phi * weight;
                  ++k;
                }
              }
            );
          }
        };
        auto inner = [&] (int poly,int i,int k,int numDofs)
        {
          // ??? assert(numDofs == innerShapeFunctionSet.size());
          //! SubVector has no size: assert(k+numDofs == localDofVector.size());
          InnerQuadratureType innerQuad( element, 2*polOrder_ );
          for (unsigned int qp=0;qp<innerQuad.nop();++qp)
          {
            auto y = innerQuad.point(qp);
            localFunction.evaluate( innerQuad[qp], value );
            double weight = innerQuad.weight(qp) *
                            element.geometry().integrationElement(y) /
                            indexSet_.volume(poly);
            innerShapeFunctionSet.evaluateTestEach(innerQuad[qp],
              [&](std::size_t alpha, typename LocalFunction::RangeType phi ) {
                int kk = alpha+k;
                //! SubVector has no size assert( kk < localDofVector.size() );
                localDofVector[ kk ] += value*phi * weight;
              }
            );
          }
        };
        apply(element,vertex,edge,inner);
      }
      // these methods are simply used to handle the vector valued case,
      // for which the interpolation needs to be applied for each component
      // separately
      template< class LocalFunction, class LocalDofVector >
      void interpolate__( const ElementType &element, const LocalFunction &localFunction,
                          LocalDofVector &localDofVector, Dune::PriorityTag<2> ) const
      {
        if constexpr (Traits::vectorSpace)
        {
          interpolate_(element,localFunction,localDofVector);
        }
        else
        {
          typedef Dune::Fem::VerticalDofAlignment<
                  ElementType, typename LocalFunction::RangeType> DofAlignmentType;
          DofAlignmentType dofAlignment(element);
          for( std::size_t i = 0; i < LocalFunction::RangeType::dimension; ++i )
          {
            Fem::Impl::SubDofVectorWrapper< LocalDofVector, DofAlignmentType > subLdv( localDofVector, i, dofAlignment );
            interpolate__(element,
                Dune::Fem::localFunctionConverter( localFunction, Fem::Impl::RangeConverter<LocalFunction::RangeType::dimension>(i) ),
                subLdv, PriorityTag<1>()
                );
          }
        }
      }
      template< class LocalFunction, class LocalDofVector >
      void interpolate__( const ElementType &element, const LocalFunction &localFunction,
                          LocalDofVector &localDofVector, Dune::PriorityTag<1> ) const
      {
        interpolate_(element,localFunction,localDofVector);
      }

      ///////////////////////////////////////////////////////////////////////////
      // interpolation onto a single intersection
      // (bool argument needed to distinguish from the method following this one)
      template< class Vertex, class Edge>
      void applyOnIntersection( const IntersectionType &intersection,
                                const Vertex &vertex, const Edge &edge,
                                Std::vector<Std::vector<unsigned int>> &mask) const
      {
        const ElementType &element = intersection.inside();
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );

        const int poly = indexSet_.index( element );
        int vertexSize, edgeOffset,edgeSize, innerOffset,innerSize;
        getSizesAndOffsets(poly, vertexSize,edgeOffset,edgeSize,innerOffset,innerSize);

        int edgeNumber = intersection.indexInInside();
        const int k = indexSet_.localIndex( element, edgeNumber, dimension-1 );
        assert(k>=0); // should only be called for 'outside' intersection
        if (k>=0)  // this doesn't make sense - remove?
        {
          std::size_t i = 0;
          if (basisSets_.vertexSize(0) > 0) //!TS
          {
            for( ; i < refElement.size( edgeNumber, dimension-1, dimension ); ++i )
            {
              int vertexNumber = refElement.subEntity( edgeNumber, dimension-1, i, dimension);
              const int vtxk = indexSet_.localIndex( element, vertexNumber, dimension );
              assert(vtxk>=0); // intersection is 'outside' so vertices should be as well
              vertex(poly,vertexNumber,i*vertexSize,1);
              for (int r=0;r<basisSets_.vectorDofs(0);++r)
                mask[0].push_back(vtxk*vertexSize+r);
              if (order2size<0>(1)>0)
              {
                assert(vertexSize==3*basisSets_.vectorDofs(0));
                for (int r=0;r<basisSets_.vectorDofs(0);++r)
                  mask[0].push_back(vtxk*vertexSize+basisSets_.vectorDofs(0)+r);
                for (int r=0;r<basisSets_.vectorDofs(0);++r)
                  mask[0].push_back(vtxk*vertexSize+2*basisSets_.vectorDofs(0)+r);
                for (int r=0;r<basisSets_.vectorDofs(0);++r)
                  mask[1].push_back(vtxk*vertexSize+basisSets_.vectorDofs(0)+r);
                for (int r=0;r<basisSets_.vectorDofs(0);++r)
                  mask[1].push_back(vtxk*vertexSize+2*basisSets_.vectorDofs(0)+r);
              }
              else
                assert(vertexSize==basisSets_.vectorDofs(0));
            }
          }
          if (order2size<1>(0)>0 ||
              order2size<1>(1)>0)
          {
            edge(poly,intersection,i,edgeSize);
            int count = 0;
            for (std::size_t alpha=0;alpha<basisSets_.edgeSize();++alpha)
              for (int deriv=0;deriv<2;++deriv)
                if (alpha < order2size<1>(deriv))
                {
                  mask[deriv].push_back(k*edgeSize+edgeOffset+count);
                  ++count;
                }
          }
        }
      }

      const IndexSetType &indexSet_;
      const BasisSetsType &basisSets_;
      const unsigned int polOrder_;
      const bool useOnb_;
    };

  } // namespace Vem

  namespace Fem
  {
    namespace Capabilities
    {
        template<class FunctionSpace, class GridPart, bool vectorSpace, bool reduced>
        struct hasInterpolation<Vem::AgglomerationVEMSpace<FunctionSpace, GridPart, vectorSpace, reduced> > {
            static const bool v = false;
        };
    }
  } // namespace Fem
} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_HK_HH
