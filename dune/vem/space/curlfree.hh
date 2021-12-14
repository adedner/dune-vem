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

    template<class GridPart>
    class CurlFreeVEMSpace;
    template< class Traits >
    class CurlFreeVEMInterpolation;

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
        , numValueShapeFunctions_(numValueSFS)
        , numGradShapeFunctions_(numGradSFS)
        , numHessShapeFunctions_(numHessSFS)
        , numInnerShapeFunctions_(innerNumSFS)
        {}

        int order () const { return sfs_.order()-1;  }

        template< class Point, class Functor >
        void evaluateEach ( const Point &xx, Functor functor ) const
        {
          // RangeType phi;
          sfs_.jacobianEach(xx, [&](std::size_t alpha, ScalarJacobianRangeType dphi)
          {
            auto x = Dune::Fem::coordinate(xx);
            if (alpha>=1)
            {
              // div: phi[0] = -dphi[0][1]; phi[1] = dphi[0][0];
              if (alpha==1) dphi[0] = {1,0};
              if (alpha==2) dphi[0] = {0,1};
              if (alpha==3) dphi[0] = {x[1],x[0]};
              if (alpha==4) dphi[0] = {2*x[0],0};
              if (alpha==5) dphi[0] = {0,2*x[1]};
              functor(alpha-1, dphi[0]);
            }
          });
        }
        template< class Point, class Functor >
        void jacobianEach ( const Point &x, Functor functor ) const
        {
          JacobianRangeType dphi;
          sfs_.hessianEach(x, [&](std::size_t alpha, ScalarHessianRangeType d2phi)
          {
            if (alpha>=dimDomain+1)
            {
              /*
              // (dx) (-dy dx) = (-dxdy dxdx)
              // (dy)            (-dydy dxdy)
              dphi[0][0] = -d2phi[0][0][1]; dphi[0][0][1] = d2phi[0][0][0];
              dphi[1][0] = -d2phi[0][1][1]; dphi[0][1][1] = d2phi[0][1][0];
              */
              functor(alpha-(dimDomain+1),d2phi[0]);
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
          assert( sfs_.order() <= 3);
          return; // !!!
          // RangeType divGrad(0);
          /*
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
          */
        }
        template< class Point, class Functor >
        void evaluateTestEach ( const Point &xx, Functor functor ) const
        {
          sfs_.jacobianEach(xx, [&](std::size_t alpha, ScalarJacobianRangeType dphi)
          {
            auto x = Dune::Fem::coordinate(xx);
            if (alpha>=1 && alpha < numInnerShapeFunctions_+1)
            {
              if (alpha==1) dphi[0] = {1,0};
              if (alpha==2) dphi[0] = {0,1};
              if (alpha==3) dphi[0] = {x[1],x[0]};
              if (alpha==4) dphi[0] = {2*x[0],0};
              if (alpha==5) dphi[0] = {0,2*x[1]};
              functor(alpha-1, dphi[0]);
            }
          });
        }

        private:
        ScalarBBBasisFunctionSetType sfs_;
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
        : intersection_(intersection), sfs_(sfs), flip_(flip)
        {}
        template< class Point, class Functor >
        void evaluateEach ( const Point &x, Functor functor ) const
        {
          // phihat = (1)
          //    phi = (d) = phihat normal
          RangeType phi;
          sfs_.evaluateEach(x, [&](std::size_t alpha, EdgeRangeType phihat)
          {
            phi = intersection_.unitOuterNormal(x);
            phi *= phihat[0]*flip_;
            functor(alpha, phi);
          });
        }
        template< class Point, class Functor >
        void jacobianEach ( const Point &x, Functor functor ) const
        {
          const auto &geo = intersection_.geometry();
          const auto &jit = geo.jacobianInverseTransposed(x);
          // dphihat = (1,1)
          //    dphi = (1,d) = J^{-T}dphihat
          //     jac = (d,d) = dphi x normal
          JacobianRangeType jac;
          sfs_.jacobianEach(x, [&](std::size_t alpha, EdgeJacobianRangeType dphihat)
          {
            DomainType normal = intersection_.unitOuterNormal(x);
            jit.mv(dphihat[0], jac[0]);
            for (int d=1;d<dimDomain;++d)
            {
              jac[d] = jac[0];
              jac[d] *= normal[d]*flip_;
            }
            jac[0] *= normal[0]*flip_;
            functor(alpha,jac);
          });
        }
        unsigned int size() const
        {
          return sfs_.size();
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
      : dofsPerCodim_(calcDofsPerCodim(order))
      , onbSFS_(Dune::GeometryType(Dune::GeometryType::cube, dimDomain), order+1) // !!!
      , edgeSFS_( Dune::GeometryType(Dune::GeometryType::cube,dimDomain-1), order )
      , useOnb_(useOnb)
      , numValueShapeFunctions_( onbSFS_.size()-1 )
      , numGradShapeFunctions_ ( onbSFS_.size()-(dimDomain+1) )
      , numHessShapeFunctions_ ( 0 ) // not implemented - needs third/forth derivatives
      , numInnerShapeFunctions_( sizeONB<0>(order+1)-1 ) // !!!!
      {
        std::cout << "[" << numValueShapeFunctions_ << ","
                  << numGradShapeFunctions_ << ","
                  << numHessShapeFunctions_ << ","
                  << numInnerShapeFunctions_ << "]"
                  << "   edge: " << edgeSFS_.size() << std::endl;
        std::cout << dofsPerCodim_[0].second << " " << dofsPerCodim_[1].second << " " << dofsPerCodim_[2].second
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
        if (intersection.neighbor())
          // !!! if (indexSet_.index(intersection.inside()) > indexSet_.index(intersection.outside()))
          if (normal[0] + normal[1] < 0)
            flip = -1;
        normal *= flip;
        // std::cout << intersection.geometry().center() << " : " << flip << " -> " << normal << std::endl;
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
        return numInnerShapeFunctions_;
      }
      int edgeValueMoments() const
      {
        return edgeSFS_.order();
      }

      std::size_t edgeSize(int deriv) const
      {
        return (deriv==0)? edgeSFS_.size() : 0;
      }

      private:
      std::array< std::pair< int, unsigned int >, dimDomain+1 > calcDofsPerCodim (unsigned int order) const
      {
        int vSize = 0;
        int eSize = sizeONB<1>(order);
        int iSize = sizeONB<0>(order+1)-1;
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
      std::array< std::pair< int, unsigned int >, dimDomain+1 > dofsPerCodim_;
      ONBShapeFunctionSetType onbSFS_;
      ScalarEdgeShapeFunctionSetType edgeSFS_;
      std::size_t numValueShapeFunctions_;
      std::size_t numGradShapeFunctions_;
      std::size_t numHessShapeFunctions_;
      std::size_t numInnerShapeFunctions_;
      bool useOnb_;
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

      typedef typename GridPartType::template Codim<0>::EntityType EntityType;

      // vem basis function sets
      typedef VEMBasisFunctionSet <EntityType, typename BasisSetsType::ShapeFunctionSetType> BasisFunctionSetType;
      typedef BasisFunctionSetType ScalarBasisFunctionSetType;

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
      using InterpolationType = CurlFreeVEMInterpolation<T>;
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
      CurlFreeVEMSpace(AgglomerationType &agglomeration,
          const unsigned int polOrder,
          int basisChoice)
      : BaseType(agglomeration, polOrder,
        typename TraitsType::BasisSetsType(polOrder, basisChoice==2),
        basisChoice, false)
      {}
    };

    //////////////////////////////////////////////////////////////////////////////
    // Interpolation classes
    //////////////////////////////////////////////////////////////////////////////

    // CurlFreeVEMInterpolation
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
    class CurlFreeVEMInterpolation
    {
      typedef CurlFreeVEMInterpolation< Traits > ThisType;

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

      static const int dimension = IndexSetType::dimension;
    private:
      typedef Dune::Fem::ElementQuadrature<GridPartType,0> InnerQuadratureType;
      typedef Dune::Fem::ElementQuadrature<GridPartType,1> EdgeQuadratureType;

      typedef typename ElementType::Geometry::ctype ctype;

    public:
      explicit CurlFreeVEMInterpolation ( const IndexSetType &indexSet,
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
        interpolate_(element,localFunction,localDofVector);
      }

      // setup right hand side constraints vector for valueProjection CLS
      // beta: current basis function phi_beta for which to setup CLS
      // volune: volume of current polygon
      // D:    Lambda(B) matrix (numDofs x numShapeFunctions)
      // d:    right hand side vector (numInnerShapeFunctions)
      template <class DomainFieldType>
      void valueL2constraints(unsigned int beta, double volume,
                              Dune::DynamicMatrix<DomainFieldType> &D,
                              Dune::DynamicVector<DomainFieldType> &d)
      {
        // !!!!!
        unsigned int numInnerShapeFunctions = d.size();
        if (numInnerShapeFunctions == 0) return;
        unsigned int numDofs = D.rows();
        for (int alpha=0; alpha<numInnerShapeFunctions; ++alpha)
        {
          if( beta - numDofs + numInnerShapeFunctions == alpha )
            d[ alpha ] = D[beta][alpha] * volume;
          else
            d[ alpha ] = 0;
        }
      }

      // fill a mask vector providing the information which dofs are
      // 'active' on the given element, i.e., are attached to a given
      // subentity of this element. Needed for dirichlet boundary data for
      // example
      void operator() ( const ElementType &element, Std::vector<char> &mask) const
      {
        std::fill(mask.begin(),mask.end(),-1);
        auto vertex = [&] (int poly,auto i,int k,int numDofs)
        {};
        auto edge = [&] (int poly,auto i,int k,int numDofs)
        {
#ifndef NDEBUG
          auto kStart = k;
#endif
          for (std::size_t alpha=0;alpha<basisSets_.edgeSize();++alpha)
          {
            if (alpha < basisSets_.edgeSize())
            {
              mask[k] = 1;
              ++k;
            }
          }
        };
        auto inner = [&mask] (int poly,auto i,int k,int numDofs)
        {
          std::fill(mask.begin()+k,mask.begin()+k+numDofs,1);
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
        {};
        auto edge = [&,this] (int poly,auto intersection,int k,int numDofs)
        { //!TS add nomral derivatives
          const auto &edgeBFS = basisSets_.edgeBasisFunctionSet( indexSet_.agglomeration(), intersection );
          int kStart = k;
          // int edgeNumber = intersection.indexInInside();
          EdgeQuadratureType edgeQuad( gridPart(), intersection, 2*polOrder_, EdgeQuadratureType::INSIDE );
          for (unsigned int qp=0;qp<edgeQuad.nop();++qp)
          {
            k = kStart;
            auto x = edgeQuad.localPoint(qp);
            auto y = intersection.geometryInInside().global(x);
            double weight = edgeQuad.weight(qp) * intersection.geometry().integrationElement(x);
            edgeBFS.evaluateEach(x,
                [&](std::size_t alpha, RangeType phi )
                {
                  basisFunctionSet.evaluateEach( y,
                    [ & ] ( std::size_t beta, typename BasisFunctionSet::RangeType value )
                    {
                      // std::cout << alpha << " " << beta << " " << k << " : "
                      //           << phi << ", " << value << std::endl;
                      assert(k<localDofMatrix.size());
                      localDofMatrix[ k ][ beta ] += value*phi * weight
                                                     / intersection.geometry().volume();
                    });
                ++k;
              });
          }
        };
        auto inner = [&] (int poly,int i,int k,int numDofs)
        {
          // ???? assert(numDofs == basisSets_.innerSize());
          InnerQuadratureType innerQuad( element, 2*polOrder_ );
          for (int qp=0;qp<innerQuad.nop();++qp)
          {
            auto y = innerQuad.point(qp);
            double weight = innerQuad.weight(qp) * element.geometry().integrationElement(y) / indexSet_.volume(poly);
            basisFunctionSet.evaluateEach( innerQuad[qp],
              [ & ] ( std::size_t beta, typename BasisFunctionSet::RangeType value )
              {
                innerShapeFunctionSet.evaluateTestEach( innerQuad[qp],
                  [&](std::size_t alpha, RangeType phi ) {
                    int kk = alpha+k;
                     std::cout << alpha << " " << beta << " " << kk << " : "
                               << phi << ", " << value << std::endl;
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

        Std::vector<std::size_t> entry(localDofVectorMatrix.size(), 0);

        auto vertex = [&] (int poly,int i,int k,int numDofs) {};
        auto edge = [&] (int poly,auto intersection,int k,int numDofs)
        {
          EdgeQuadratureType edgeQuad( gridPart(),
                intersection, 2*polOrder_, EdgeQuadratureType::INSIDE );
          for (unsigned int qp=0;qp<edgeQuad.nop();++qp)
          {
            auto x = edgeQuad.localPoint(qp);
            auto xx = x;
            double weight = edgeQuad.weight(qp) * intersection.geometry().integrationElement(x);
            edgeShapeFunctionSet.evaluateEach( x, [ & ] ( std::size_t beta, typename EdgeShapeFunctionSet::RangeType value ) {
                edgeBFS.evaluateEach( xx,
                  [&](std::size_t alpha, typename EdgeShapeFunctionSet::RangeType phi ) {
                    assert( entry[0]+alpha < localDofVectorMatrix[0].size() );
                    localDofVectorMatrix[0][ entry[0]+alpha ][ beta ] += value*phi * weight
                                                                         / intersection.geometry().volume();
                  });
              }
            );
          }
          entry[0] += edgeBFS.size();
        };
        applyOnIntersection(intersection,vertex,edge,mask);
        assert( entry[0] == localDofVectorMatrix[0].size() );
        assert( entry[1] == localDofVectorMatrix[1].size() );

        if (localDofVectorMatrix[0].size() > 0)
        {
          try
          {
            localDofVectorMatrix[0].invert();
          }
          catch (const FMatrixError&)
          {
            std::cout << "Interpolation: localDofVectorMatrix[0].invert() failed!\n";
            const auto &M = localDofVectorMatrix[0];
            for (std::size_t alpha=0;alpha<M.size();++alpha)
            {
              for (std::size_t beta=0;beta<M[alpha].size();++beta)
                std::cout << M[alpha][beta] << " ";
              std::cout << std::endl;
            }
            assert(0);
            throw FMatrixError();
          }
        }
      }

    private:

      void getSizesAndOffsets(int poly,
                  int &vertexSize,
                  int &edgeOffset, int &edgeSize,
                  int &innerOffset, int &innerSize) const
      {
        auto dofs   = basisSets_.dofsPerCodim();  // assume always three entries in dim order (i.e. 2d)
        assert(dofs.size()==3);
        vertexSize  = dofs[0].second;
        edgeOffset  = indexSet_.subAgglomerates(poly,dimension)*vertexSize;
        edgeSize    = dofs[1].second;
        innerOffset = edgeOffset + indexSet_.subAgglomerates(poly,dimension-1)*edgeSize;
        innerSize   = dofs[2].second;
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
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );

        // use the bb set for this polygon for the inner testing space
        const auto &innerShapeFunctionSet = basisSets_.basisFunctionSet( indexSet_.agglomeration(), element );

        // define the vertex,edge, and inner parts of the interpolation
        auto vertex = [&] (int poly,auto i,int k,int numDofs) {};
        auto edge = [&] (int poly,auto intersection,int k,int numDofs)
        { //!TS edge derivatives
          const auto &edgeBFS = basisSets_.edgeBasisFunctionSet( indexSet_.agglomeration(), intersection );
          int kStart = k;
          EdgeQuadratureType edgeQuad( gridPart(),
                intersection, 2*polOrder_, EdgeQuadratureType::INSIDE );
          for (unsigned int qp=0;qp<edgeQuad.nop();++qp)
          {
            k = kStart;
            auto x = edgeQuad.localPoint(qp);
            auto y = intersection.geometryInInside().global(x);
            localFunction.evaluate( y, value );
            double weight = edgeQuad.weight(qp) * intersection.geometry().integrationElement(x);
            edgeBFS.evaluateEach(x,
              [&](std::size_t alpha, typename LocalFunction::RangeType phi ) {
                localDofVector[ k ] += value*phi * weight
                                       / intersection.geometry().volume();
                ++k;
              });
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
            double weight = innerQuad.weight(qp) * element.geometry().integrationElement(y) / indexSet_.volume(poly);
            innerShapeFunctionSet.evaluateTestEach(innerQuad[qp],
              [&](std::size_t alpha, typename LocalFunction::RangeType phi ) {
                localDofVector[ alpha+k ] += value*phi * weight;
              }
            );
          }
        };
        apply(element,vertex,edge,inner);
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
          edge(poly,intersection,i,edgeSize);
          int count = 0;
          for (std::size_t alpha=0;alpha<basisSets_.edgeSize(0);++alpha)
          {
            mask[0].push_back(k*edgeSize+edgeOffset+count);
            ++count;
          }
        }
      }

      const IndexSetType &indexSet_;
      BasisSetsType basisSets_;
      const unsigned int polOrder_;
      const bool useOnb_;
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

#endif // #ifndef DUNE_VEM_SPACE_HK_HH
