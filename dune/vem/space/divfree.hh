#ifndef DUNE_VEM_SPACE_DIVFREE_HH
#define DUNE_VEM_SPACE_DIVFREE_HH

#include <cassert>
#include <utility>

#include <dune/common/dynmatrix.hh>
#include <dune/grid/common/exceptions.hh>

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
#include <dune/vem/space/interpolation.hh>
#include <dune/vem/space/default.hh>

namespace Dune
{
  namespace Vem
  {
    // Internal Forward Declarations
    // -----------------------------

    template<class GridPart>
    struct DivFreeVEMSpace;

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
      static constexpr bool valReduced = false;  // true: only project values into k-1 polynomials
      static constexpr bool jacReduced = false;  // true: only use grad phi_ for gradient basisfunctions
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
        , scale_( 1 ) // std::sqrt( sfs_.bbox().volume() ) )
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
              phi[0] *= scale_;
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
          sfs_.bbox().gradientTransform(y, true);
          y *= std::sqrt( sfs_.bbox().volume() );
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
              dphi[0] *= scale_;
              functor(alpha-1+numInnerShapeFunctions_, dphi[0]);
              ++test;
            }
          });
          if (test != numValueShapeFunctions_)
            std::cout << "evaluated " << test << " basis functions instead of " << numValueShapeFunctions_ << std::endl;
          assert(test == numValueShapeFunctions_);
        }
        template< class Point, class Functor >
        void jacValEach ( const Point &x, Functor functor ) const
        {
          /*
          RangeType y = sfs_.position( x );
          sfs_.bbox().gradientTransform(y, true);
          y *= std::sqrt( sfs_.bbox().volume() );
          assert( y.two_norm() < 1.5 );
          // D ( -y phi )  = ( - y phi_x         - phi - y phi_y )
          //   (  x phi )    ( phi + x phi_x     x phi_y         )
          sfs_.jacobianEach(x, [&](std::size_t alpha, ScalarRangeType phi)
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
              dphi[0] *= scale_;
              functor(alpha-1+numInnerShapeFunctions_, dphi[0]);
              ++test;
            }
          });
          */
          DUNE_THROW( NotImplemented, "DivFree does not implement the improved gradient projection yet");
        }

        template< class Point, class Functor >
        void jacobianEach ( const Point &x, Functor functor ) const
        {
          if constexpr (!jacReduced)
          {
            JacobianRangeType jac(0);
            sfs_.evaluateEach(x, [&](std::size_t alpha, ScalarRangeType phi)
            {
              if (alpha*dimDomain*dimDomain < numGradShapeFunctions_)
              {
                for (size_t d1=0;d1<dimDomain;++d1)
                {
                  for (size_t d2=0;d2<dimDomain;++d2)
                  {
                    jac[d1][d2] = phi[0];
                    functor(alpha*dimDomain*dimDomain+d1*dimDomain+d2, jac);
                    jac[d1][d2] = 0;
                  }
                }
              }
            });
          }
          else
          {
            vsfs_.jacobianEach(x, [&](std::size_t alpha, JacobianRangeType dphi)
            {
              if (alpha>=dimRange) functor(alpha-dimRange,dphi);
            });
          }
        }

        template< class Point, class Functor >
        void hessianEach ( const Point &x, Functor functor ) const
        {}
        // functor(alpha, psi) with psi in R^r
        //
        // for each g = g_{alpha*D+s*D+t} = m_alpha e_s e_t           (1<=alpha<=numGradSF and 1<=s,t<=dimDomain)
        // sum_{ij} int_E d_j v_i g_ij = - sum_{ij} int_E v_i d_j g_ij + ...
        //     = - int_E sum_i ( v_i sum_j d_j g_ij )
        //     = - int_E v . psi
        // with psi=(psi_i) and psi_i = sum_j d_j g_ij
        //
        // g_{ij} = m_a delta_{js} delta_{it}   (m=m_alpha and fixed s=1,..,dimDomain)
        // psi_i = sum_j d_j g_ij = sum_j d_j m_a delta_{js} delta_{it}
        //                        = d_s m_a delta_t
        template< class Point, class Functor >
        void divJacobianEach( const Point &x, Functor functor ) const
        {
          RangeType divGrad(0);
          if constexpr (!jacReduced)
          {
            sfs_.jacobianEach(x, [&](std::size_t alpha, ScalarJacobianRangeType dphi)
            {
              if (alpha*dimDomain*dimDomain < numGradShapeFunctions_)
              {
                for (size_t d1=0;d1<dimDomain;++d1)
                {
                  for (size_t d2=0;d2<dimDomain;++d2)
                  {
                    divGrad[d1] = dphi[0][d2];
                    functor(alpha*dimDomain*dimDomain+d1*dimDomain+d2, divGrad);
                    divGrad[d1] = 0;
                  }
                }
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
          sfs_.bbox().gradientTransform(y, true);
          y *= std::sqrt( sfs_.bbox().volume() );
          assert( y.two_norm() < 1.5 );
          sfs_.evaluateEach(x, [&](std::size_t alpha, ScalarRangeType phi)
          {
            if (alpha < numInnerShapeFunctions_)
            {
              RangeType val{-y[1]*phi[0], y[0]*phi[0]};
              functor(alpha, val );
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
                             bool twist,
                             unsigned int numEdgeTestFunctions)
        : intersection_(intersection), sfs_(sfs), twist_(twist), numEdgeTestFunctions_(numEdgeTestFunctions)
        {}
        template< class Point, class Functor >
        void evaluateEach ( const Point &xx, Functor functor ) const
        {
          auto x = Dune::Fem::coordinate(xx);
          if (twist_) x[0] = 1.-x[0];
          sfs_.evaluateEach(x,functor);
        }
        template< class Point, class Functor >
        void jacobianEach ( const Point &xx, Functor functor ) const
        {
          auto x = Dune::Fem::coordinate(xx);
          if (twist_) x[0] = 1.-x[0];
          JacobianRangeType jac;
          const auto &geo = intersection_.geometry();
          const auto &jit = geo.jacobianInverseTransposed(x);
          sfs_.jacobianEach(x, [&](std::size_t alpha, EdgeJacobianRangeType dphi)
          {
            if (twist_) dphi *= -1;
            for (std::size_t r=0;r<dimRange;++r)
              jit.mv(dphi[r],jac[r]);
            functor(alpha,jac);
          });
        }
        template< class Point, class Functor >
        void evaluateTestEach ( const Point &xx, Functor functor ) const
        {
          auto x = Dune::Fem::coordinate(xx);
          if (twist_) x[0] = 1.-x[0];
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
        bool twist_;
      };

      public:
      typedef ShapeFunctionSet ShapeFunctionSetType;
      typedef EdgeShapeFunctionSet EdgeShapeFunctionSetType;

      DivFreeVEMBasisSets( const int order, bool useOnb, bool conforming )
      : innerOrder_( order )
      , conforming_( conforming )
      , onbSFS_(Dune::GeometryType(Dune::GeometryType::cube, dimDomain),
          valReduced? order-1:order+1
        )
      , edgeSFS_( Dune::GeometryType(Dune::GeometryType::cube,dimDomain-1), maxEdgeDegree() )
      , dofsPerCodim_(calcDofsPerCodim(order))
      , useOnb_(useOnb)
      , numValueShapeFunctions_( sizeONB<0>(onbSFS_.order()-1) )
      , numGradShapeFunctions_ (
              (!jacReduced)?
              sizeONB<0>(order-1)*BBBasisFunctionSetType::RangeType::dimension:
              (onbSFS_.size()-1)*BBBasisFunctionSetType::RangeType::dimension
          )
      , numHessShapeFunctions_ ( 0 )
      , numInnerShapeFunctions_( order==2 ? 0 :
                                 sizeONB<0>(order-3)/BBBasisFunctionSetType::RangeType::dimension )
      , numOrthoShapeFunctions_(valReduced?
          numInnerShapeFunctions_:
          sizeONB<0>(order-1)/BBBasisFunctionSetType::RangeType::dimension
        )
      , numEdgeTestShapeFunctions_( (conforming_)? sizeONB<1>(order-2):sizeONB<1>(order-1) )
      {
        auto degrees = edgeDegrees();
        /*
        std::cout << "dofsPerCodim:" << dofsPerCodim_[0].second << " "
                                     << dofsPerCodim_[1].second << " "
                                     << dofsPerCodim_[2].second << std::endl;
        std::cout << "[" << numValueShapeFunctions_ << ","
                  << numGradShapeFunctions_ << ","
                  << numHessShapeFunctions_ << ","
                  << numInnerShapeFunctions_ << "]"
                  << constraintSize() << "]"
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
        */
      }

      const std::size_t maxOrder() const
      {
        return onbSFS_.order();
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
             const Agglomeration &agglomeration,
             const IntersectionType &intersection, bool twist) const
      {
        return EdgeShapeFunctionSetType(intersection, edgeSFS_, 0, numEdgeTestShapeFunctions_);
      }
      std::size_t size( std::size_t orderSFS ) const
      {
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
        if (conforming_)
        {
          return numInnerShapeFunctions_ + onbSFS_.size()-1;
        }
        else
        {
          // innerOrder_ fails (P^3 solution not exactly interpolated with innerOrder_==3
          return numInnerShapeFunctions_ +
                 sizeONB<0>(innerOrder_-1) / BBBasisFunctionSetType::RangeType::dimension
                 -1;
        }
      }
      std::size_t vertexSize(int deriv) const
      {
        // vertex values in div free space
        if (conforming_)
        {
          if (deriv==0)
            return pow(dimDomain,deriv);
          else
            return 0;
        }
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
        if (conforming_)
          return innerOrder_-2;
        else
          return innerOrder_-1;
          // return innerOrder_;
      }
      std::size_t edgeSize(int deriv) const
      {
        auto degrees = edgeDegrees();
        return degrees[deriv] < 0 ? 0 : sizeONB<1>( degrees[deriv] );
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
        {
          if (conforming_)
            return pow(dimDomain,deriv);
          else
            return 0;
        }
        if (dim == 1 && deriv == 0)
        {
          if (conforming_)
          {
            if (innerOrder_-2<0)
              return 0;
            else
              return Dune::Fem::OrthonormalShapeFunctions<1>::size(innerOrder_-2);
          }
          else
          {
            if (innerOrder_-1<0)
              return 0;
            else
              return Dune::Fem::OrthonormalShapeFunctions<1>::size(innerOrder_-1);
          /*
            if (innerOrder_<0)
              return 0;
            else
              return Dune::Fem::OrthonormalShapeFunctions<1>::size(innerOrder_);
           */
          }
        }
        if (dim == 2 && deriv == 0 && innerOrder_ >=3)
          return Dune::Fem::OrthonormalShapeFunctions<2>::size(innerOrder_-3);
        else
          return 0;
      }
      int vectorDofs(int dim) const
      {
        // return 1 for scalar dofs (inner moments) or dimRange for vector dofs (vertex and edge)
        // BBBasisFunctionSetType::FunctionSpaceType::RangeType::dimension = dimRange
        if (dim == 2)
          return 1;
        if (dim == 0 || dim == 1)
          return BBBasisFunctionSetType::FunctionSpaceType::RangeType::dimension;
        else // shouldn't get here
        {
          assert(false);
          return -1;
        }
      }

      private:
      Std::vector<int> edgeDegrees() const
      {
        Std::vector<int> degrees(2, -1);
        if (conforming_)
        {
          degrees[0] += 2;
          degrees[0] += std::max(0,innerOrder_-1);
        }
        else
          degrees[0] = innerOrder_-1;
          // degrees[0] = innerOrder_;
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
        // std::cout << "vSize: " << vSize << std::endl;
        auto eSize = order2size<1>(0) * BBBasisFunctionSetType::RangeType::dimension;
        // std::cout << "eSize: " << eSize << std::endl;
        auto iSize = order2size<2>(0);
        // std::cout << "iSize: " << iSize << std::endl;
        return std::array< std::pair< int, unsigned int >, dimDomain+1 >
               { std::make_pair( dimDomain,   vSize ),
                 std::make_pair( dimDomain-1, eSize ),
                 std::make_pair( dimDomain-2, iSize ) };
      }

      // note: the actual shape function set depends on the entity so
      // we can only construct the underlying monomial basis in the ctor
      const int innerOrder_;
      const bool useOnb_;
      const bool conforming_;
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
      friend struct DivFreeVEMSpace<GridPart>;
      typedef DivFreeVEMSpace<GridPart> DiscreteFunctionSpaceType;

      static const int codimension = 0;
      static const int dimDomain = FunctionSpaceType::DomainType::dimension;
      static const int dimRange = FunctionSpaceType::RangeType::dimension;
      static const bool vectorSpace = true;
      typedef Hybrid::IndexRange<int, 1> LocalBlockIndices;
      // static const int baseRangeDimension = dimRange;

      typedef typename GridPartType::template Codim<codimension>::EntityType EntityType;


      // types for the mapper
      typedef VemAgglomerationIndexSet <GridPartType> IndexSetType;
    };

    // DivFreeVEMSpace
    // ---------------------
    template<class GridPart>
    struct DivFreeVEMSpace : public DefaultAgglomerationVEMSpace<
          AgglomerationVEMSpaceTraits<DivFreeVEMSpaceTraits<GridPart>,
            double>, long double >
    {
      typedef AgglomerationVEMSpaceTraits<DivFreeVEMSpaceTraits<GridPart>,
              double> TraitsType;
      typedef DefaultAgglomerationVEMSpace<TraitsType,
              long double> BaseType;
      typedef typename BaseType::AgglomerationType AgglomerationType;
      typedef typename BaseType::BasisSetsType::ShapeFunctionSetType::FunctionSpaceType FunctionSpaceType;
      typedef typename BaseType::DomainFieldType DomainFieldType;
      DivFreeVEMSpace(AgglomerationType &agglomeration,
          const unsigned int polOrder,
          const bool conforming,
          int basisChoice,
          bool edgeInterpolation)
      : BaseType(agglomeration,polOrder,
                 typename TraitsType::BasisSetsType(polOrder, basisChoice, conforming),
                 basisChoice,edgeInterpolation)
      {
        // TODO: move this to the default and add a method to the baisisSets to
        // obtain the required order (here polOrder+1)
        assert( BaseType::basisSets_.maxOrder() == polOrder+1 );
        BaseType::update(true);
      }

    protected:
      virtual void setupConstraintRHS(const Std::vector<Std::vector<typename BaseType::ElementSeedType> > &entitySeeds, unsigned int agglomerate,
                                      typename BaseType::ComputeMatrixType &RHSconstraintsMatrix, double volume) override
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
                  = BaseType::basisSets_.edgeBasisFunctionSet(BaseType::agglomeration(),
                               intersection, BaseType::blockMapper().indexSet().twist(intersection) );

            Std::vector<Std::vector<unsigned int>> mask(2,Std::vector<unsigned int>(0)); // contains indices with Phi_mask[i] is attached to given edge
            edgePhiVector[0] = 0;
            edgePhiVector[1] = 0;

            BaseType::interpolation()(intersection, edgeShapeFunctionSet, edgePhiVector, mask);

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
                /*
                else
                {
                  std::cout << "shouldn't get here!\n";
                  std::cout << "should have " << alpha << "<" << RHSconstraintsMatrix[0].size() << std::endl;
                  abort();
                }
                */
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
