#ifndef DUNE_VEM_SPACE_INTERPOLATION_HH
#define DUNE_VEM_SPACE_INTERPOLATION_HH

#include <cstddef>

#include <initializer_list>
#include <utility>

#include <dune/common/dynmatrix.hh>
#include <dune/geometry/referenceelements.hh>

// #include <dune/fem/space/localfiniteelement/interpolation.hh>
#include <dune/fem/function/localfunction/converter.hh>
#include <dune/fem/space/combinedspace/interpolation.hh>

#include <dune/fem/quadrature/elementquadrature.hh>
#include <dune/fem/space/shapefunctionset/orthonormal.hh>
// #include <dune/fem/space/shapefunctionset/legendre.hh>
#include <dune/vem/agglomeration/basisfunctionset.hh>
// #include <dune/vem/agglomeration/shapefunctionset.hh>
// #include <dune/vem/misc/highorderquadratures.hh>
#include <dune/vem/misc/vector.hh>

namespace Dune
{
  namespace Vem
  {

    template <class Traits>
    struct AgglomerationVEMTestBasisSets
    {
      AgglomerationVEMTestBasisSets( const unsigned int order0, const unsigned int order1, bool useOnb)
      : scalarSFS_(Dune::GeometryType(Dune::GeometryType::cube, Traits::dimension),order0)
      , edgeSFS_( Dune::GeometryType(Dune::GeometryType::cube,Traits::dimension-1), order1 )
      , useOnb_(useOnb)
      {}
      template <class Agglomeration>
      const typename Traits::BBBasisFunctionSetType bbBasisFunctionSet(
             const Agglomeration &agglomeration, const typename Traits::EntityType &entity) const
      {
        const std::size_t agglomerate = agglomeration.index(entity);
        return typename Traits::BBBasisFunctionSetType( entity, agglomerate, agglomeration.boundingBoxes(), useOnb_, scalarSFS_);
      }
      template <class Agglomeration>
      const typename Traits::EdgeShapeFunctionSetType &edgeBasisFunctionSet(
             const Agglomeration &agglomeration, const typename Traits::EntityType &entity) const
      {
        return edgeSFS_;
      }
      std::size_t size() const
      {
        return scalarSFS_.size()*Traits::baseRangeDimension;
      }
      std::size_t edgeSize() const
      {
        return edgeSFS_.size();
      }
      private:
      // note: the actual shape function set depends on the entity so
      // we can only construct the underlying monomial basis in the ctor
      typename Traits::ScalarShapeFunctionSetType scalarSFS_;
      typename Traits::EdgeShapeFunctionSetType edgeSFS_;
      bool useOnb_;
    };

    // AgglomerationVEMInterpolation
    // -----------------------------



    template< class Traits >
    class AgglomerationVEMInterpolation
    {
      typedef AgglomerationVEMInterpolation< Traits > ThisType;

    public:
      typedef typename Traits::AgglomerationIndexSetType AgglomerationIndexSetType;
      typedef typename AgglomerationIndexSetType::ElementType ElementType;
      typedef typename AgglomerationIndexSetType::GridPartType GridPartType;
      typedef typename GridPartType::IntersectionType IntersectionType;

      static const int dimension = AgglomerationIndexSetType::dimension;
      static const int baseRangeDimension = Traits::baseRangeDimension;
    private:
#if 1 // FemQuads
      typedef Dune::Fem::ElementQuadrature<GridPartType,0> InnerQuadratureType;
      typedef Dune::Fem::ElementQuadrature<GridPartType,1> EdgeQuadratureType;
#else
      typedef Dune::Fem::ElementQuadrature<GridPartType,0,Dune::Fem::HighOrderQuadratureTraits> InnerQuadratureType;
      typedef Dune::Fem::ElementQuadrature<GridPartType,1,Dune::Fem::HighOrderQuadratureTraits> EdgeQuadratureType;
#endif
      typedef typename ElementType::Geometry::ctype ctype;
      typedef typename Traits::BBBasisFunctionSetType InnerShapeFunctionSet;

    public:
      explicit AgglomerationVEMInterpolation ( const AgglomerationIndexSetType &indexSet, unsigned int polOrder, bool useOnb ) noexcept
        : indexSet_( indexSet )
        , testBasisSets_( std::max(indexSet_.maxDegreePerCodim()[2],0),
                          std::max(indexSet_.maxDegreePerCodim()[1],0),
                          useOnb )
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
      void operator() ( const ElementType &element, Std::vector<char> &mask) const
      {
        std::fill(mask.begin(),mask.end(),-1);
        auto vertex = [&] (int poly,auto i,int k,int numDofs)
        {
          for (int r=0;r<baseRangeDimension;++r)
            mask[k+r] = 1;
          k += baseRangeDimension;
          if (indexSet_.template order2size<0>(1)>0)
          {
            for (int r=0;r<baseRangeDimension;++r)
            {
              mask[k+2*r]   = 2;
              mask[k+2*r+1] = 2;
            }
          }
        };
        auto edge = [&] (int poly,auto i,int k,int numDofs)
        {
#ifndef NDEBUG
          auto kStart = k;
#endif
          for (std::size_t alpha=0;alpha<testBasisSets_.edgeSize();++alpha)
          {
            if (alpha < indexSet_.template order2size<1>(0))
            {
              mask[k] = 1;
              ++k;
            }
            if (alpha < indexSet_.template order2size<1>(1))
            {
              mask[k] = 2;
              ++k;
            }
          }
          assert(k-kStart == numDofs);
        };
        auto inner = [&mask] (int poly,auto i,int k,int numDofs)
        { std::fill(mask.begin()+k,mask.begin()+k+numDofs,1); };
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
        const auto &innerShapeFunctionSet = testBasisSets_.bbBasisFunctionSet( indexSet_.agglomeration(), element );
        const auto &edgeBFS = testBasisSets_.edgeBasisFunctionSet( indexSet_.agglomeration(), element );

        // define the corresponding vertex,edge, and inner parts of the interpolation
        auto vertex = [&] (int poly,int i,int k,int numDofs)
        { //!TS add derivatives at vertex for conforming space
          const auto &x = refElement.position( i, dimension );
          basisFunctionSet.evaluateEach( x, [ &localDofMatrix, k ] ( std::size_t alpha, typename BasisFunctionSet::RangeType phi ) {
              assert( phi.dimension == baseRangeDimension );
              if (alpha < localDofMatrix[k].size())
                for (int r=0;r<phi.dimension;++r)
                  localDofMatrix[ k+r ][ alpha ] = phi[ r ];
            } );
          k += baseRangeDimension;
          if (indexSet_.template order2size<0>(1)>0)
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
            double weight = edgeQuad.weight(qp) * intersection.geometry().integrationElement(x);
            edgeBFS.evaluateEach(x,
                [&](std::size_t alpha, typename BasisFunctionSet::RangeType phi ) {
                if (alpha < indexSet_.template order2size<1>(0))
                {
                  basisFunctionSet.evaluateEach( y,
                    [ & ] ( std::size_t beta, typename BasisFunctionSet::RangeType value )
                    {
                      assert(k<localDofMatrix.size());
                      localDofMatrix[ k ][ beta ] += value*phi * weight
                                                     / intersection.geometry().volume();
                    }
                  );
                  ++k;
                }
                if (alpha < indexSet_.template order2size<1>(1))
                {
                  basisFunctionSet.jacobianEach( y,
                    [ & ] ( std::size_t beta, typename BasisFunctionSet::JacobianRangeType dvalue )
                    {
                      // we assume here that jacobianEach is in global
                      // space so the jit is not applied
                      assert(k<localDofMatrix.size());
                      typename BasisFunctionSet::RangeType dn;
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
          assert(numDofs == innerShapeFunctionSet.size());
          InnerQuadratureType innerQuad( element, 2*polOrder_ );
          for (int qp=0;qp<innerQuad.nop();++qp)
          {
            auto y = innerQuad.point(qp);
            double weight = innerQuad.weight(qp) * element.geometry().integrationElement(y) / indexSet_.volume(poly);
            basisFunctionSet.evaluateEach( innerQuad[qp],
              [ & ] ( std::size_t beta, typename BasisFunctionSet::RangeType value )
              {
                innerShapeFunctionSet.evaluateEach( innerQuad[qp],
                  [&](std::size_t alpha, typename InnerShapeFunctionSet::RangeType phi ) {
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
      //!TS what do we want to return here?
      //!TS  i) std::vector<DynMatrix> containing the matrix for each normal derivative?
      //!TS ii) add an int normDerivOrder parameter and fill the matrix only for that order?
      template< class EdgeShapeFunctionSet >
      void operator() (const IntersectionType &intersection,
                       const EdgeShapeFunctionSet &edgeShapeFunctionSet, Std::vector < Dune::DynamicMatrix<double> > &localDofVectorMatrix,
                       Std::vector<Std::vector<unsigned int>> &mask) const
      {
        for (std::size_t i=0;i<mask.size();++i)
          mask[i].clear();
        const ElementType &element = intersection.inside();
        const auto &edgeBFS = testBasisSets_.edgeBasisFunctionSet( indexSet_.agglomeration(), element );
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
          const auto &x = edgeGeo.local( refElement.position( i, dimension ) );
          edgeShapeFunctionSet.evaluateEach( x, [ &localDofVectorMatrix, &entry ] ( std::size_t alpha, typename EdgeShapeFunctionSet::RangeType phi ) {
              assert( phi.dimension == baseRangeDimension );
              assert( entry[0] < localDofVectorMatrix[0].size() );
              if (alpha < localDofVectorMatrix[0][ entry[0] ].size())
                for (int r=0;r<phi.dimension;++r)
                  localDofVectorMatrix[0][ entry[0]+r ][ alpha ] = phi[ r ];
            } );
          entry[0] += baseRangeDimension;
          if (indexSet_.template order2size<0>(1)>0)
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
            entry[0] += baseRangeDimension;
            entry[1] += baseRangeDimension;
          }
        };
        auto edge = [&] (int poly,auto intersection,int k,int numDofs)
        { //!TS add normal derivatives
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
                    //!TS add alpha<...
                    if (alpha < indexSet_.template order2size<1>(0) &&
                        beta < indexSet_.edgeSize(0))
                    {
                      assert( entry[0]+alpha < localDofVectorMatrix[0].size() );
                      localDofVectorMatrix[0][ entry[0]+alpha ][ beta ] += value*phi * weight
                                                                           / intersection.geometry().volume();
                    }
                    // FIX ME
                    if (alpha < indexSet_.template order2size<1>(1) &&
                        beta < indexSet_.edgeSize(1))
                    {
                      assert( entry[1]+alpha < localDofVectorMatrix[1].size() );
                      localDofVectorMatrix[1][ entry[1]+alpha ][ beta ] += value*phi * weight * flipNormal;
                    }
                  }
                );
              }
            );
          }
          entry[0] += indexSet_.template order2size<1>(0)*baseRangeDimension;
          entry[1] += indexSet_.template order2size<1>(1)*baseRangeDimension;
        };
        applyOnIntersection(intersection,vertex,edge,mask);
        assert( entry[0] == localDofVectorMatrix[0].size() );
        assert( entry[1] == localDofVectorMatrix[1].size() );
      }

    private:
      void getSizesAndOffsets(int poly,
                  int &vertexSize,
                  int &edgeOffset, int &edgeSize,
                  int &innerOffset, int &innerSize) const
      {
        auto dofs   = indexSet_.dofsPerCodim();  // assume always three entries in dim order (i.e. 2d)
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

        // vertex dofs
        //!TS needs changing
        if (indexSet_.vertexOrders()[0] >= 0)
        {
          for( int i = 0; i < refElement.size( dimension ); ++i )
          {
            const int k = indexSet_.localIndex( element, i, dimension) * vertexSize;
            if ( k >= 0 ) // is a 'real' vertex of the polygon
              vertex(poly,i,k,1);
          }
        }
        //!TS needs changing
        if (indexSet_.template order2size<1>(0)>0 ||
            indexSet_.template order2size<1>(1)>0)
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
        if (indexSet_.innerOrders()[0] >=0)
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
        assert( value.dimension == baseRangeDimension );
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );

        // use the bb set for this polygon for the inner testing space
        const auto &innerShapeFunctionSet = testBasisSets_.bbBasisFunctionSet( indexSet_.agglomeration(), element );
        const auto &edgeBFS = testBasisSets_.edgeBasisFunctionSet( indexSet_.agglomeration(), element );

        // define the vertex,edge, and inner parts of the interpolation
        auto vertex = [&] (int poly,auto i,int k,int numDofs)
        { //!TS vertex derivatives
          const auto &x = refElement.position( i, dimension );
          localFunction.evaluate( x, value );
          //! SubDofWrapper does not have size assert( k < localDofVector.size() );
          for (int r=0;r<value.dimension;++r)
            localDofVector[ k+r ] = value[ r ];
          k += value.dimension;
          if (indexSet_.template order2size<0>(1)>0)
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
            if (indexSet_.template order2size<1>(1)>0)
              localFunction.jacobian( y, dvalue);
            typename LocalFunction::RangeType dnvalue;
            dvalue.mv(normal,dnvalue);
            assert( dnvalue[0] == dvalue[0]*normal );
            edgeBFS.evaluateEach(x,
              [&](std::size_t alpha, typename LocalFunction::RangeType phi ) {
                //! SubDofWrapper has no size assert( kk < localDofVector.size() );
                if (alpha < indexSet_.template order2size<1>(0))
                {
                  //! see above assert(k < localDofVector.size() );
                  localDofVector[ k ] += value*phi * weight
                                         / intersection.geometry().volume();
                  ++k;
                }
                if (alpha < indexSet_.template order2size<1>(1))
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
          assert(numDofs == innerShapeFunctionSet.size());
          //! SubVector has no size: assert(k+numDofs == localDofVector.size());
          InnerQuadratureType innerQuad( element, 2*polOrder_ );
          for (unsigned int qp=0;qp<innerQuad.nop();++qp)
          {
            auto y = innerQuad.point(qp);
            localFunction.evaluate( innerQuad[qp], value );
            double weight = innerQuad.weight(qp) * element.geometry().integrationElement(y) / indexSet_.volume(poly);
            innerShapeFunctionSet.evaluateEach(innerQuad[qp],
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
          if (indexSet_.vertexOrders()[0]>=0) //!TS
          {
            for( ; i < refElement.size( edgeNumber, dimension-1, dimension ); ++i )
            {
              int vertexNumber = refElement.subEntity( edgeNumber, dimension-1, i, dimension);
              const int vtxk = indexSet_.localIndex( element, vertexNumber, dimension );
              assert(vtxk>=0); // intersection is 'outside' so vertices should be as well
              vertex(poly,vertexNumber,i*vertexSize,1);
              mask[0].push_back(vtxk*vertexSize);
              if (indexSet_.template order2size<0>(1)>0)
              {
                assert(vertexSize==3);
                mask[0].push_back(vtxk*vertexSize+1);
                mask[0].push_back(vtxk*vertexSize+2);
                mask[1].push_back(vtxk*vertexSize+1);
                mask[1].push_back(vtxk*vertexSize+2);
              }
              else
                assert(vertexSize==1);
            }
          }
          if (indexSet_.template order2size<1>(0)>0 ||
              indexSet_.template order2size<1>(1)>0)
          {
            edge(poly,intersection,i,edgeSize);
            int count = 0;
            for (std::size_t alpha=0;alpha<testBasisSets_.edgeSize();++alpha)
              for (int deriv=0;deriv<2;++deriv)
                if (alpha < indexSet_.template order2size<1>(deriv))
                {
                  mask[deriv].push_back(k*edgeSize+edgeOffset+count);
                  ++count;
                }
          }
        }
      }

      const AgglomerationIndexSetType &indexSet_;
      AgglomerationVEMTestBasisSets<Traits> testBasisSets_;
      const unsigned int polOrder_;
      const bool useOnb_;
    };
  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_INTERPOLATION_HH
