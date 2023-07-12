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
#include <dune/fem/space/common/localrestrictprolong.hh>
#include <dune/fem/space/shapefunctionset/orthonormal.hh>
#include <dune/fem/function/localfunction/converter.hh>
#include <dune/fem/space/combinedspace/interpolation.hh>
#include <dune/vem/misc/compatibility.hh>

#include <dune/vem/agglomeration/basisfunctionset.hh>
#include <dune/vem/misc/vector.hh>
#include <dune/vem/space/interpolation.hh>
#include <dune/vem/space/default.hh>

namespace Dune
{
  namespace Vem
  {
    // Internal Forward Declarations
    // -----------------------------

    template<class FunctionSpace, class GridPart,
             bool vectorSpace, bool reduced,
             class StorageField, class ComputeField>
    struct AgglomerationVEMSpace;

    // IsAgglomerationVEMSpace
    // -----------------------

    template<class FunctionSpace, class GridPart, bool vectorSpace, bool reduced, class StorageField, class ComputeField >
    struct IsAgglomerationVEMSpace<AgglomerationVEMSpace<FunctionSpace, GridPart,
           vectorSpace,reduced,StorageField,ComputeField> >
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
        void jacValEach ( const Point &x, Functor functor ) const
        {
          sfs_.jacobianEach(x, functor);
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
          sfs_.jacobianEach(x, [&](std::size_t alpha, EdgeJacobianRangeType dphi)
          {
            functor(alpha,dphi);
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

      static int chooseOrder(int defaultOrder, int userOrder)
      { return userOrder<0? defaultOrder: userOrder; }
      AgglomerationVEMBasisSets( const std::size_t order,
                                 const std::array<int,3> orderTuple,
                                 const TestSpacesType &testSpaces,
                                 int basisChoice )
      // use order2size
      : testSpaces_(testSpaces)
      , useOnb_(basisChoice == 2)
      , dofsPerCodim_(calcDofsPerCodim())
      , maxOrder_( chooseOrder( std::max(order,maxEdgeDegree()),orderTuple[0] ) )
      , onbSFS_(Dune::GeometryType(Dune::GeometryType::cube, dimDomain), maxOrder_)
      , edgeSFS_( Dune::GeometryType(Dune::GeometryType::cube,dimDomain-1), maxEdgeDegree() )
      , numValueShapeFunctions_( onbSFS_.size()*BBBasisFunctionSetType::RangeType::dimension )
      , numGradShapeFunctions_ (
          !reduced? std::min( numValueShapeFunctions_,
                              sizeONB<0>( chooseOrder(maxOrder_-1,orderTuple[1]) ) )
          : numValueShapeFunctions_-1*BBBasisFunctionSetType::RangeType::dimension
        )
      , numHessShapeFunctions_ (
          !reduced? std::min( numValueShapeFunctions_,
                              sizeONB<0>( chooseOrder(maxOrder_-2,orderTuple[2]) ) )
          : 0 // numValueShapeFunctions_-3*BBBasisFunctionSetType::RangeType::dimension
        )
      , numInnerShapeFunctions_( testSpaces[2][0]<0? 0 : sizeONB<0>(testSpaces[2][0]) )
      , numEdgeTestShapeFunctions_( sizeONB<1>(
                 *std::max_element( testSpaces_[1].begin(), testSpaces_[1].end()) ) )
      {
        auto degrees = edgeDegrees();
        /*
        std::cout << "order=" << order << " using " << maxOrder_ << ": "
                  << "[" << numValueShapeFunctions_ << ","
                  << numGradShapeFunctions_ << ","
                  << numHessShapeFunctions_ << ","
                  << constraintSize() << ","
                  << numInnerShapeFunctions_ << "]"
                  << "   edge: ["
                  << edgeSize(0) << "," << edgeSize(1) << ","
                  << numEdgeTestShapeFunctions_ << "]"
                  << " " << degrees[0] << " " << degrees[1]
                  << " max size of edge set: " << edgeSFS_.size()
                  << std::endl;
        */
      }

      const std::size_t maxOrder() const
      {
        return maxOrder_;
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
             const Agglomeration &agglomeration, const IntersectionType
             &intersection, bool twist) const
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
      static std::size_t sizeONB(int order)
      {
        if (order<0) return 0;
        else return Dune::Fem::OrthonormalShapeFunctions<dimDomain - codim> :: size(order) *
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
      const std::size_t maxOrder_;
      const ONBShapeFunctionSetType onbSFS_;
      const ScalarEdgeShapeFunctionSetType edgeSFS_;
      const std::size_t numValueShapeFunctions_;
      const std::size_t numGradShapeFunctions_;
      const std::size_t numHessShapeFunctions_;
      const std::size_t numInnerShapeFunctions_;
      const std::size_t numEdgeTestShapeFunctions_;
    };



    template<class FunctionSpace, class GridPart,
             bool vectorspace, bool reduced, class StorageField, class ComputeField>
    struct HkSpaceTraits
    {
      typedef AgglomerationVEMBasisSets<FunctionSpace,GridPart,vectorspace,reduced> BasisSetsType;

      static const bool vectorSpace = vectorspace;
      friend struct AgglomerationVEMSpace<FunctionSpace, GridPart, vectorSpace, reduced, StorageField, ComputeField>;

      typedef AgglomerationVEMSpace<FunctionSpace, GridPart, vectorSpace, reduced, StorageField, ComputeField> DiscreteFunctionSpaceType;

      typedef GridPart GridPartType;

      static const int dimension = GridPartType::dimension;
      static const int codimension = 0;
      static const int dimDomain = FunctionSpace::DomainType::dimension;
      static const int dimRange = FunctionSpace::RangeType::dimension;
      typedef Hybrid::IndexRange<int, dimRange> LocalBlockIndices;

      typedef typename GridPartType::template Codim<codimension>::EntityType EntityType;
      typedef VemAgglomerationIndexSet <GridPartType> IndexSetType;
      typedef FunctionSpace FunctionSpaceType;
    };

    // AgglomerationVEMSpace
    // ---------------------
    template<class FunctionSpace, class GridPart,
             bool vectorSpace, bool reduced,
             class StorageField, class ComputeField>
    struct AgglomerationVEMSpace
    : public DefaultAgglomerationVEMSpace<
             AgglomerationVEMSpaceTraits<HkSpaceTraits<FunctionSpace,GridPart,vectorSpace,reduced,StorageField,ComputeField>,
                                         StorageField>,
             ComputeField>
    {
      typedef AgglomerationVEMSpaceTraits<HkSpaceTraits<FunctionSpace,GridPart,vectorSpace,reduced,StorageField,ComputeField>,
                                          StorageField> TraitsType;
      typedef DefaultAgglomerationVEMSpace<TraitsType,
               ComputeField> BaseType;
      typedef Agglomeration<GridPart> AgglomerationType;
      typedef typename TraitsType::FunctionSpaceType FunctionSpaceType;
      typedef typename BaseType::DomainFieldType DomainFieldType;
      AgglomerationVEMSpace(AgglomerationType &agglomeration,
          const unsigned int polOrder,
          const std::array<int,3> orderTuple,
          const typename TraitsType::BasisSetsType::TestSpacesType &testSpaces,
          int basisChoice,
          bool edgeInterpolation)
      : BaseType(agglomeration,polOrder,
                 typename TraitsType::BasisSetsType(polOrder, orderTuple,testSpaces, basisChoice),
                 basisChoice,edgeInterpolation)
      {
        BaseType::update(true);
      }

    protected:
      virtual void setupConstraintRHS(const Std::vector<Std::vector<typename BaseType::ElementSeedType> > &entitySeeds,
                                      unsigned int agglomerate,
                                      typename BaseType::ComputeMatrixType &RHSconstraintsMatrix,
                                      double volume) override
      {
        //////////////////////////////////////////////////////////////////////////
        /// Fix RHS constraints for value projection /////////////////////////////
        //////////////////////////////////////////////////////////////////////////

        static constexpr int dimRange = TraitsType::dimRange;
        static constexpr int blockSize = TraitsType::vectorSpace ? dimRange : 1;
        // const std::size_t numShapeFunctions = BaseType::basisSets_.size(0);
        const std::size_t numDofs = BaseType::blockMapper().numDofs(agglomerate) * blockSize;
        const std::size_t numConstraintShapeFunctions = BaseType::basisSets_.constraintSize();
        const std::size_t numInnerShapeFunctions = BaseType::basisSets_.innerSize();
        const std::size_t numConstraints = RHSconstraintsMatrix.cols();
        int polOrder = BaseType::order();

        assert( numDofs == RHSconstraintsMatrix.rows() );
        assert( numConstraintShapeFunctions <= numConstraints );
        assert( numInnerShapeFunctions <= numConstraintShapeFunctions );
        if (numConstraints == 0) return;

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

        if (RHSconstraintsMatrix[0].size() == numInnerShapeFunctions )
          return;

        // only case covered is triangles with C^1-conf space of order 2 with extra constraint
        assert(BaseType::basisSets_.edgeSize(1)>0 && numConstraints == numConstraintShapeFunctions+1);
        std::size_t alpha = numConstraints-1;
        assert( alpha == numConstraintShapeFunctions );
        // matrices for edge projections
        Std::vector<Dune::DynamicMatrix<double> > edgePhiVector(2);

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

            double flipNormal = 1.;
            if (intersection.neighbor()) // we need to check the orientation of the normal
              if ( BaseType::indexSet().index(intersection.inside()) >
                   BaseType::indexSet().index(intersection.outside()) )
                flipNormal = -1;

            const typename BaseType::BasisSetsType::EdgeShapeFunctionSetType edgeShapeFunctionSet
                  = BaseType::basisSets_.edgeBasisFunctionSet(BaseType::agglomeration(),
                               intersection, BaseType::blockMapper().indexSet().twist(intersection) );

            Std::vector<Std::vector<unsigned int>> mask(2,Std::vector<unsigned int>(0)); // contains indices with Phi_mask[i] is attached to given edge
            edgePhiVector[0].resize(BaseType::basisSets_.edgeSize(0),
                                    BaseType::basisSets_.edgeSize(0), 0);
            edgePhiVector[1].resize(BaseType::basisSets_.edgeSize(1),
                                    BaseType::basisSets_.edgeSize(1), 0);

            BaseType::interpolation()(intersection, edgeShapeFunctionSet, edgePhiVector, mask);

            // now compute int_e Phi^e m_alpha
            typename BaseType::Quadrature1Type quadrature(BaseType::gridPart(), intersection, 2 * polOrder + 1, BaseType::Quadrature1Type::INSIDE);
            for (std::size_t qp = 0; qp < quadrature.nop(); ++qp)
            {
              auto x = quadrature.localPoint(qp);
              auto y = intersection.geometryInInside().global(x);
              const DomainFieldType weight = intersection.geometry().integrationElement(x) * quadrature.weight(qp);
              auto normal = intersection.unitOuterNormal(x);
              edgeShapeFunctionSet.evaluateEach(x, [&](std::size_t beta,
                        typename BaseType::BasisSetsType::EdgeShapeFunctionSetType::RangeType psi)
              {
                if (beta < edgePhiVector[1].size())
                  for (std::size_t s=0; s<mask[1].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                    RHSconstraintsMatrix[mask[1][s]][alpha] += weight *
                                         edgePhiVector[1][beta][s] * psi;
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
        template<class FunctionSpace, class GridPart, bool vectorSpace, bool reduced, class StorageField, class ComputeField>
        struct hasInterpolation<Vem::AgglomerationVEMSpace<FunctionSpace, GridPart, vectorSpace, reduced, StorageField, ComputeField> > {
            static const bool v = false;
        };
    }
    template<class FunctionSpace, class GridPart, bool vectorSpace, bool reduced, class StorageField, class ComputeField>
    class DefaultLocalRestrictProlong <
        Vem::AgglomerationVEMSpace<FunctionSpace, GridPart, vectorSpace, reduced, StorageField, ComputeField> >
    : public EmptyLocalRestrictProlong<
        Vem::AgglomerationVEMSpace<FunctionSpace, GridPart, vectorSpace, reduced, StorageField, ComputeField> >
    {
      typedef EmptyLocalRestrictProlong<
        Vem::AgglomerationVEMSpace<FunctionSpace, GridPart, vectorSpace, reduced, StorageField, ComputeField> > BaseType;
      public:
      DefaultLocalRestrictProlong( const Vem::AgglomerationVEMSpace<FunctionSpace, GridPart, vectorSpace, reduced, StorageField, ComputeField> &space )
        : BaseType()
      {}
    };

  } // namespace Fem
} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_HK_HH
