#ifndef DUNE_VEM_SPACE_DIVFREE_HH
#define DUNE_VEM_SPACE_DIVFREE_HH

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
    class DivFreeVEMSpace;

    template<class GridPart>
    struct IsAgglomerationVEMSpace<DivFreeVEMSpace<GridPart> >
            : std::integral_constant<bool, true> {
    };

    // DivFreeVEMSpaceTraits
    // ---------------------------

    template<class FunctionSpace, class GridPart, bool reduced=true>
    struct DivFreeVEMBasisSets
    {
      typedef GridPart GridPartType;
      static constexpr bool vectorSpace = true;
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

    private:
      struct ShapeFunctionSet
      {
        typedef typename ScalarFunctionSpaceType::RangeType ScalarRangeType;
        typedef typename ScalarFunctionSpaceType::JacobianRangeType ScalarJacobianRangeType;
        typedef typename ScalarFunctionSpaceType::HessianRangeType ScalarHessianRangeType;

        typedef typename BBBasisFunctionSetType::FunctionSpaceType FunctionSpaceType;
        typedef typename FunctionSpaceType::RangeType RangeType;
        typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;
        typedef typename FunctionSpaceType::HessianRangeType HessianRangeType;
        static const int dimDomain = FunctionSpaceType::DomainType::dimension;
        static const int dimRange = RangeType::dimension;

        ShapeFunctionSet() = default;
        template <class Agglomeration>
        ShapeFunctionSet(bool useOnb, const ONBShapeFunctionSetType& onbSFS,
                         std::size_t numValueSF, std::size_t numGradSF, std::size_t numHessSF,
                         std::size_t numInnerSF, std::size_t numOrthoSF,
                         const Agglomeration &agglomeration, const EntityType &entity)
        : vsfs_(entity, agglomeration.index(entity),
                agglomeration.boundingBoxes(), useOnb, onbSFS)
        , sfs_(entity, agglomeration.index(entity),
               agglomeration.boundingBoxes(), useOnb, onbSFS)
        , entity_(entity)
        , scale_( std::sqrt( sfs_.bbox().volume() ) )
        , numValueShapeFunctions_(numValueSF)
        , numGradShapeFunctions_(numGradSF)
        , numHessShapeFunctions_(numHessSF)
        , numInnerShapeFunctions_(numInnerSF)
        , numOrthoShapeFunctions_(numOrthoSF)
        {}

        int order () const { return sfs_.order()-1;  }

        const auto &valueBasisSet() const
        {
          return *this;
        }

        template< class Point, class Functor >
        void scalarEach ( const Point &x, Functor functor ) const
        {
          sfs_.evaluateEach(x, [&](std::size_t alpha, ScalarRangeType phi)
          {
            // TODO check what condition on alpha needed here
            // scalarEach used in RHS constraints set up
            if (alpha>=1)
            {
              // phi[0] *= scale_;
              functor(alpha-1, phi[0]);
            }
          });
        }
        /*
             ortho   1     x    y        scalar   x     y    xy    x^2   y^2
        k=2                                      (1)   (0)
                                                 (0)   (1)
        k=3         (-y)                         (1)   (0)   (y)   (2x)  (0)
                    ( x)                         (0)   (1)   (x)   (0)   (2y)
        */
        template< class Point, class Functor >
        void evaluateEach ( const Point &x, Functor functor ) const
        {
          // Note: basis functions that are included in the constraint have
          // to be evaluated first and unconstraint basis functions have to
          // be together at the end.
          // To achieve this the 'ortho' part of the basis set is split
          // with the 'alpha' for the 'inner' once being 0,..,numInner-1
          // and the additional 'ortho' basisfunctions getting 'alpha'
          // values so that a gap is left for the 'grad' basisfunctions.
          int test = 0;
          RangeType y = sfs_.position( x );
          assert( y.two_norm() < 1.5 );
          sfs_.evaluateEach(x, [&](std::size_t alpha, ScalarRangeType phi)
          {
            if (alpha < numOrthoShapeFunctions_)
            {
              //// (-y,x) * phi(x,y)
              RangeType val{-y[1]*phi[0], y[0]*phi[0]};
              if (alpha<numInnerShapeFunctions_)
                functor(alpha,val);
              else
                functor(alpha+sfs_.size()-1,val);
              ++test;
            }
          });
          sfs_.jacobianEach(x, [&](std::size_t alpha, ScalarJacobianRangeType dphi)
          {
            if (alpha>=1)
            {
              // dphi[0] *= scale_;
              functor(alpha-1+numInnerShapeFunctions_, dphi[0]);
              ++test;
            }
          });
          assert(test == numValueShapeFunctions_);
        }

        template< class Point, class Functor >
        void jacobianEach ( const Point &x, Functor functor ) const
        {}

        template< class Point, class Functor >
        void hessianEach ( const Point &x, Functor functor ) const
        {}
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
            vsfs_.jacobianEach(x, [&](std::size_t alpha, JacobianRangeType dphi)
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
            vsfs_.hessianEach(x, [&](std::size_t alpha, HessianRangeType d2phi)
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
        }

        template< class Point, class Functor >
        void evaluateTestEach ( const Point &x, Functor functor ) const
        {
          RangeType y = sfs_.position( x );
          sfs_.evaluateEach(x, [&](std::size_t alpha, ScalarRangeType phi)
          {
            if (alpha < numInnerShapeFunctions_)
            {
              functor(alpha, {-y[1]*phi[0], y[0]*phi[0]} );
            }
          });
        }

        private:
        BBBasisFunctionSetType vsfs_;
        ScalarBBBasisFunctionSetType sfs_;
        EntityType entity_;
        double scale_;
        std::size_t numValueShapeFunctions_;
        std::size_t numGradShapeFunctions_;
        std::size_t numHessShapeFunctions_;
        std::size_t numInnerShapeFunctions_;
        std::size_t numOrthoShapeFunctions_;
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

        typedef Fem::VectorialShapeFunctionSet<ScalarEdgeShapeFunctionSetType, RangeType> VectorEdgeShapeFunctionSetType;
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

      DivFreeVEMBasisSets( const int order, bool useOnb )
      : innerOrder_( order )
      // test order+1 later
      , onbSFS_(Dune::GeometryType(Dune::GeometryType::cube, dimDomain), order+1)
      , edgeSFS_( Dune::GeometryType(Dune::GeometryType::cube,dimDomain-1), maxEdgeDegree() )
      , dofsPerCodim_(calcDofsPerCodim(order))
      , useOnb_(useOnb)
      , numValueShapeFunctions_( sizeONB<0>(onbSFS_.order()-1) )
      , numGradShapeFunctions_ ( 0 )
         // !reduced? std::min( numValueShapeFunctions_, sizeONB<0>(std::max(0, order - 1)) )
         // : numValueShapeFunctions_-1*BBBasisFunctionSetType::RangeType::dimension )
      , numHessShapeFunctions_ ( 0 )
      , numInnerShapeFunctions_( order==2 ? 0 :
                                 sizeONB<0>(order-3)/BBBasisFunctionSetType::RangeType::dimension )
      , numOrthoShapeFunctions_( sizeONB<0>(order-1)/BBBasisFunctionSetType::RangeType::dimension )
      , numEdgeTestShapeFunctions_( sizeONB<1>(order-2) )
      {
        auto degrees = edgeDegrees();
        std::cout << "dofsPerCodim:" << dofsPerCodim_[0].second << " "
                                     << dofsPerCodim_[1].second << " "
                                     << dofsPerCodim_[2].second << std::endl;
        std::cout << "[" << numValueShapeFunctions_ << ","
                  << numGradShapeFunctions_ << ","
                  << numHessShapeFunctions_ << ","
                  << numInnerShapeFunctions_ << "]"
                  << "   edge: ["
                  << edgeSize(0) << "," << edgeSize(1) << ","
                  << numEdgeTestShapeFunctions_ << "]"
                  << " " << degrees[0] << " " << degrees[1]
                  << " max size of edge set: " << edgeSFS_.size()*2
                  << " edgeSize(): " << edgeSize()
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
        return ShapeFunctionSet(useOnb_, onbSFS_, numValueShapeFunctions_, numGradShapeFunctions_, numHessShapeFunctions_,
                                numInnerShapeFunctions_,numOrthoShapeFunctions_,
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
      std::size_t constraintSize() const
      {
        return numInnerShapeFunctions_ + onbSFS_.size()-1;
        return numInnerShapeFunctions_; // add for div in P_0 constraints: + onbSFS_.size()-1;
        return sizeONB<0>(innerOrder_-2);
      }
      std::size_t vertexSize(int deriv) const
      {
        // vertex values in div free space
        if (deriv==0)
          return pow(dimDomain,deriv);
        else
          return 0;
      }
      std::size_t innerSize() const
      {
        return numInnerShapeFunctions_;
      }
      std::size_t edgeValueMoments() const
      {
        // returns order of edge moments up to P_k where k is the entry in dof tuple
        return innerOrder_-2;
      }
      std::size_t edgeSize(int deriv) const
      {
        auto degrees = edgeDegrees();
        return degrees[deriv] < 0 ? 0 : sizeONB<1>( degrees[deriv] );
        /* Dune::Fem::OrthonormalShapeFunctions<1>::size( degrees[deriv] )
           * BBBasisFunctionSetType::RangeType::dimension; */
      }

      ////////////////////////////
      // used in interpolation  //
      ////////////////////////////
      std::size_t edgeSize() const
      {
        return edgeSFS_.size() * BBBasisFunctionSetType::RangeType::dimension;
      }
      std::size_t numEdgeTestShapeFunctions() const
      {
        return numEdgeTestShapeFunctions_;
      }
      template <int dim>
      std::size_t order2size(unsigned int deriv) const
      {
        if (dim == 0 && deriv == 0) // vertex size
          return pow(dimDomain,deriv);
        if (dim == 1 && deriv == 0)
        {
          // assert(  == edgeSize(deriv) );
          if (innerOrder_-2<0)
            return 0;
          else
            return Dune::Fem::OrthonormalShapeFunctions<1>::size(innerOrder_-2);
        }
        if (dim == 2 && deriv == 0 && innerOrder_ >=3)
          return Dune::Fem::OrthonormalShapeFunctions<2>::size(innerOrder_-3);
        else
          return 0;
      }

      private:
      Std::vector<int> edgeDegrees() const
      {
        // std::array<std::vector<int>,dimDomain+1> testSpaces;

        // testSpaces[0].resize(2,-1);
        // testSpaces[1].resize(2,-1);
        // testSpaces[2].resize(2,-1);

        // testSpaces[0][0] = 0;
        // testSpaces[1][0] = order-2;
        // testSpaces[2][0] = order-2;

        // assert( testSpaces_[2].size()<2 );

        Std::vector<int> degrees(2, -1);
        degrees[0] += 2;
        degrees[0] += std::max(0,innerOrder_-1);

        // for (std::size_t i=0;i<testSpaces_[0].size();++i)
        //   degrees[i] += 2*(testSpaces_[0][i]+1);
        // if (testSpaces_[0].size()>1 && testSpaces_[0][1]>-1) // add tangential derivatives
        //   degrees[0] += 2;
        // for (std::size_t i=0;i<testSpaces_[1].size();++i)
        //   degrees[i] += std::max(0,testSpaces_[1][i]+1);
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

      std::array< std::pair< int, unsigned int >, dimDomain+1 > calcDofsPerCodim (unsigned int order) const
      {
        auto vSize = order2size<0>(0) * BBBasisFunctionSetType::RangeType::dimension;
        std::cout << "vSize: " << vSize << std::endl;
        auto eSize = order2size<1>(0) * BBBasisFunctionSetType::RangeType::dimension;
        std::cout << "eSize: " << eSize << std::endl;
        auto iSize = order2size<2>(0);
        std::cout << "iSize: " << iSize << std::endl;
        return std::array< std::pair< int, unsigned int >, dimDomain+1 >
               { std::make_pair( dimDomain,   vSize ),
                 std::make_pair( dimDomain-1, eSize ),
                 std::make_pair( dimDomain-2, iSize ) };
      }

      // note: the actual shape function set depends on the entity so
      // we can only construct the underlying monomial basis in the ctor
      const int innerOrder_;
      const bool useOnb_;
      std::array< std::pair< int, unsigned int >, dimDomain+1 > dofsPerCodim_;
      const ONBShapeFunctionSetType onbSFS_;
      const ScalarEdgeShapeFunctionSetType edgeSFS_;
      const std::size_t numValueShapeFunctions_;
      const std::size_t numGradShapeFunctions_;
      const std::size_t numHessShapeFunctions_;
      const std::size_t numInnerShapeFunctions_;
      const std::size_t numOrthoShapeFunctions_;
      const std::size_t numEdgeTestShapeFunctions_;
    };



    template<class GridPart>
    struct DivFreeVEMSpaceTraits
    {
      typedef GridPart GridPartType;
      static const int dimension = GridPartType::dimension;
      typedef Dune::Fem::FunctionSpace<double,double,dimension,dimension> FunctionSpaceType;

      typedef DivFreeVEMBasisSets<FunctionSpaceType,GridPart> BasisSetsType;
      friend class DivFreeVEMSpace<GridPart>;
      typedef DivFreeVEMSpace<GridPart> DiscreteFunctionSpaceType;

      static const int codimension = 0;
      static const int dimDomain = FunctionSpaceType::DomainType::dimension;
      static const int dimRange = FunctionSpaceType::RangeType::dimension;
      static const bool vectorSpace = true;
      static const int baseRangeDimension = dimRange;

      typedef typename GridPartType::template Codim<codimension>::EntityType EntityType;

      // vem basis function sets
      typedef VEMBasisFunctionSet <EntityType, typename BasisSetsType::ShapeFunctionSetType> ScalarBasisFunctionSetType;
      typedef std::conditional_t< vectorSpace,
              ScalarBasisFunctionSetType,
              Fem::VectorialBasisFunctionSet<ScalarBasisFunctionSetType, typename FunctionSpaceType::RangeType>
              > BasisFunctionSetType;

      // types for the mapper
      typedef Hybrid::IndexRange<int, 1> LocalBlockIndices;
      typedef VemAgglomerationIndexSet <GridPartType> IndexSetType;
      typedef AgglomerationDofMapper <GridPartType, IndexSetType> BlockMapperType;

      template<class DiscreteFunction, class Operation = Fem::DFCommunicationOperation::Copy>
      struct CommDataHandle {
          typedef Operation OperationType;
          typedef Fem::DefaultCommunicationHandler <DiscreteFunction, Operation> Type;
      };

      template <class T>
      using InterpolationType = AgglomerationVEMInterpolation<T>;
    };

    // DivFreeVEMSpace
    // ---------------------
    template<class GridPart>
    struct DivFreeVEMSpace
    : public DefaultAgglomerationVEMSpace< DivFreeVEMSpaceTraits<GridPart> >
    {
      typedef DivFreeVEMSpaceTraits<GridPart> TraitsType;
      typedef DefaultAgglomerationVEMSpace<TraitsType> BaseType;
      typedef typename BaseType::AgglomerationType AgglomerationType;
      typedef typename BaseType::BasisSetsType::ShapeFunctionSetType::FunctionSpaceType FunctionSpaceType;
      typedef typename FunctionSpaceType::DomainFieldType DomainFieldType;
      DivFreeVEMSpace(AgglomerationType &agglomeration,
          const unsigned int polOrder,
          int basisChoice,
          bool edgeInterpolation)
      : BaseType(agglomeration,polOrder,
                 typename TraitsType::BasisSetsType(polOrder, basisChoice),
                 basisChoice,edgeInterpolation)
      {
        // TODO: move this to the default and add a method to the baisisSets to
        // obtain the required order (here polOrder+1)
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
        const std::size_t dimDomain = DomainType::dimension;
        const std::size_t dimRange = RangeType::dimension;
        static constexpr int blockSize = BaseType::localBlockSize;
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
              RHSconstraintsMatrix[ beta ][ alpha ] = volume;
          }
        }

        /*
          q in P_{k-1}\R
          int_E v grad q = - int_E div(v) q + int_bE q v.n
                         = - div(v) int_E q + int_bE q v.n
                         = int_bE q v.n
          since int_E q = int_E 1 q = 0 since q is from ONB set
        */

#if 1
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
                alpha += numInnerShapeFunctions;
                if (alpha<RHSconstraintsMatrix[0].size())
                {
                  edgeShapeFunctionSet.evaluateEach(x, [&](std::size_t beta,
                        typename BaseType::BasisSetsType::EdgeShapeFunctionSetType::RangeType psi)
                  {
                    for (std::size_t s=0; s<mask[0].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                      // put into correct offset place in constraint RHS matrix
                      RHSconstraintsMatrix[mask[0][s]][alpha] += weight * edgePhiVector[0][beta][s] * psi*normal * m;
                  });
                }
                else
                {
                  std::cout << "shouldn't get here!\n";
                  std::cout << "should have " << alpha << "<" << RHSconstraintsMatrix[0].size() << std::endl;
                  abort();
                }
              });
            } // quadrature loop
          } // loop over intersections
        } // loop over triangles in agglomerate
#endif
      }
    };

  } // namespace Vem

  namespace Fem
  {
    namespace Capabilities
    {
        template<class GridPart>
        struct hasInterpolation<Vem::DivFreeVEMSpace<GridPart> > {
            static const bool v = false;
        };
    }
  } // namespace Fem
} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_DIVFREE_HH
