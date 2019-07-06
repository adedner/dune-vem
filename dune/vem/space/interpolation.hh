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
#include <dune/vem/agglomeration/basisfunctionset.hh>
// #include <dune/vem/agglomeration/shapefunctionset.hh>

namespace Dune
{
  namespace Vem
  {

    // AgglomerationVEMInterpolation
    // -----------------------------

    template< class AgglomerationIndexSet >
    class AgglomerationVEMInterpolation
    {
      typedef AgglomerationVEMInterpolation< AgglomerationIndexSet > ThisType;

    public:
      typedef typename AgglomerationIndexSet::ElementType ElementType;
      typedef typename AgglomerationIndexSet::GridPartType GridPartType;
      typedef typename GridPartType::IntersectionType IntersectionType;

    private:
      typedef Dune::Fem::ElementQuadrature<GridPartType,0> InnerQuadratureType;
      typedef Dune::Fem::ElementQuadrature<GridPartType,1> EdgeQuadratureType;
      typedef typename ElementType::Geometry::ctype ctype;
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld-1,1> EdgeFSType;
      typedef Dune::Fem::OrthonormalShapeFunctionSet<EdgeFSType> EdgeShapeFunctionSetType;
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld,1> InnerFSType;
      typedef Dune::Fem::OrthonormalShapeFunctionSet<InnerFSType> InnerShapeFunctionSetType;

    public:
      explicit AgglomerationVEMInterpolation ( const AgglomerationIndexSet &indexSet, unsigned int polOrder ) noexcept
        : indexSet_( indexSet )
        , edgeBFS_(  Dune::GeometryType(Dune::GeometryType::cube,GridPartType::dimension-1), std::max(indexSet_.testSpaces()[1],0) )
        , innerBFS_( Dune::GeometryType(Dune::GeometryType::cube,GridPartType::dimension),   std::max(indexSet_.testSpaces()[2],0) )
        , polOrder_( polOrder )
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
      void operator() ( const ElementType &element, std::vector<bool> &mask) const
      {
        auto set = [&mask] (int poly,auto i,int k,int numDofs)
        { std::fill(mask.begin()+k,mask.begin()+k+numDofs,true); };
        apply(element,set,set,set);
      }

      // preform interpolation of a full shape function set filling a transformation matrix
      template< class ShapeFunctionSet, class LocalDofMatrix >
      void operator() ( const BoundingBoxBasisFunctionSet< GridPartType, ShapeFunctionSet > &shapeFunctionSet,
                       LocalDofMatrix &localDofMatrix ) const
      {
        const int dimension = AgglomerationIndexSet::dimension;
        const ElementType &element = shapeFunctionSet.entity();
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );

        // use the bb set for this polygon for the inner testing space
        const auto &bbox = indexSet_.boundingBox(element);
        BoundingBoxBasisFunctionSet< GridPartType, InnerShapeFunctionSetType > innerShapeFunctionSet( element, bbox, innerBFS_ );

        // define the corresponding vertex,edge, and inner parts of the interpolation
        auto vertex = [&] (int poly,int i,int k,int numDofs)
        {
          const auto &x = refElement.position( i, dimension );
          shapeFunctionSet.evaluateEach( x, [ &localDofMatrix, k ] ( std::size_t alpha, typename ShapeFunctionSet::RangeType phi ) {
              assert( phi.dimension == 1 );
              localDofMatrix[ k ][ alpha ] = phi[ 0 ];
            } );
        };
        auto edge = [&] (int poly,auto intersection,int k,int numDofs)
        {
          assert(numDofs == edgeBFS_.size());
          int edgeNumber = intersection.indexInInside();
          EdgeQuadratureType edgeQuad( gridPart(), intersection, 2*polOrder_, EdgeQuadratureType::INSIDE );
          for (int qp=0;qp<edgeQuad.nop();++qp)
          {
            auto x = edgeQuad.localPoint(qp);
            auto y = intersection.geometryInInside().global(x);
            double weight = edgeQuad.weight(qp) * intersection.geometry().integrationElement(x) / intersection.geometry().volume();
            shapeFunctionSet.evaluateEach( y,
              [ & ] ( std::size_t beta, typename ShapeFunctionSet::RangeType value )
              {
                edgeBFS_.evaluateEach( x,
                  [&](std::size_t alpha, typename EdgeFSType::RangeType phi ) {
                    int kk = alpha+k;
                    assert(kk<localDofMatrix.size());
                    localDofMatrix[ kk ][ beta ] += value[0]*phi[0] * weight;
                  }
                );
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
            shapeFunctionSet.evaluateEach( innerQuad[qp],
              [ & ] ( std::size_t beta, typename ShapeFunctionSet::RangeType value )
              {
                innerShapeFunctionSet.evaluateEach( innerQuad[qp],
                  [&](std::size_t alpha, typename InnerFSType::RangeType phi ) {
                    int kk = alpha+k;
                    assert(kk<localDofMatrix.size());
                    localDofMatrix[ kk ][ beta ] += value[0]*phi[0] * weight;
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
      template< class ShapeFunctionSet >
      void operator() (const IntersectionType &intersection,
                       const ShapeFunctionSet &shapeFunctionSet, Dune::DynamicMatrix<double> &localDofMatrix,
                       std::vector<int> &mask) const
      {
        const int dimension = AgglomerationIndexSet::dimension;
        mask.clear();
        const ElementType &element = intersection.inside();
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );
        int edgeNumber = intersection.indexInInside();
        const auto &edgeGeo = refElement.template geometry<1>(edgeNumber);

        // define the three relevant part of the interpolation, i.e.,
        // vertices,edges - no inner needed since only doing interpolation
        // on intersection
        auto vertex = [&] (int poly,int i,int k,int numDofs)
        {
          const auto &x = edgeGeo.local( refElement.position( i, dimension ) );
          shapeFunctionSet.evaluateEach( x, [ &localDofMatrix, k ] ( std::size_t alpha, typename ShapeFunctionSet::RangeType phi ) {
              assert( phi.dimension == 1 );
              assert( k < localDofMatrix.size() );
              localDofMatrix[ k ][ alpha ] = phi[ 0 ];
            } );
        };
        auto edge = [&] (int poly,auto intersection,int k,int numDofs)
        {
          assert(numDofs == edgeBFS_.size());
          EdgeQuadratureType edgeQuad( gridPart(),
                intersection, 2*polOrder_, EdgeQuadratureType::INSIDE );
          for (int qp=0;qp<edgeQuad.nop();++qp)
          {
            auto x = edgeQuad.localPoint(qp);
            auto xx = x;
            double weight = edgeQuad.weight(qp) * intersection.geometry().integrationElement(x) / intersection.geometry().volume();
            shapeFunctionSet.evaluateEach( x, [ & ] ( std::size_t beta, typename ShapeFunctionSet::RangeType value ) {
                edgeBFS_.evaluateEach( xx,
                  [&](std::size_t alpha, typename EdgeFSType::RangeType phi ) {
                    int kk = alpha+k;
                    assert( kk < localDofMatrix.size() );
                    localDofMatrix[ kk ][ beta ] += value[0]*phi[0] * weight;
                  }
                );
              }
            );
          }
        };
        applyOnIntersection(intersection,vertex,edge,mask);
      }

    private:
      void getSizesAndOffsets(int poly,
                  int &vertexSize,
                  int &edgeOffset, int &edgeSize,
                  int &innerOffset, int &innerSize) const
      {
        const int dimension = AgglomerationIndexSet::dimension;
        auto dofs   = indexSet_.dofsPerCodim();  // assume always three entries in dim order (i.e. 2d)
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
        const int dimension = AgglomerationIndexSet::dimension;
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );
        const int poly = indexSet_.index( element );
        int vertexSize, edgeOffset,edgeSize, innerOffset,innerSize;
        getSizesAndOffsets(poly, vertexSize,edgeOffset,edgeSize,innerOffset,innerSize);

        // vertex dofs
        if (indexSet_.testSpaces()[0] >= 0)
        {
          for( int i = 0; i < refElement.size( dimension ); ++i )
          {
            const int k = indexSet_.localIndex( element, i, dimension );
            if ( k >= 0 ) // is a 'real' vertex of the polygon
              vertex(poly,i,k,1);
          }
        }
        if (indexSet_.testSpaces()[1] >= 0)
        {
          // to avoid any issue with twists we use an intersection iterator
          // here instead of going over the edges
          auto it = gridPart().ibegin( element );
          const auto endit = gridPart().iend( element );
          for( ; it != endit; ++it )
          {
            const auto& intersection = *it;
            const int i = intersection.indexInInside();
            const int k = indexSet_.localIndex( element, i, dimension-1 )*edgeSize + edgeOffset;
            if ( k>=edgeOffset ) // 'real' edge of polygon
              edge(poly,intersection,k,edgeSize);
          }
        }
        if (indexSet_.testSpaces()[2] >=0)
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
        const int dimension = AgglomerationIndexSet::dimension;
        typename LocalFunction::RangeType value;
        assert( value.dimension == 1 );
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );

        // use the bb set for this polygon for the inner testing space
        const auto &bbox = indexSet_.boundingBox(element);
        BoundingBoxBasisFunctionSet< GridPartType, InnerShapeFunctionSetType > innerShapeFunctionSet( element, bbox, innerBFS_ );

        // define the vertex,edge, and inner parts of the interpolation
        auto vertex = [&] (int poly,auto i,int k,int numDofs)
        {
          const auto &x = refElement.position( i, dimension );
          localFunction.evaluate( x, value );
          //! SubDofWrapper does not have size assert( k < localDofVector.size() );
          localDofVector[ k ] = value[ 0 ];
        };
        auto edge = [&] (int poly,auto intersection,int k,int numDofs)
        {
          assert(numDofs == edgeBFS_.size());
          int edgeNumber = intersection.indexInInside();
          EdgeQuadratureType edgeQuad( gridPart(),
                intersection, 2*polOrder_, EdgeQuadratureType::INSIDE );
          for (int qp=0;qp<edgeQuad.nop();++qp)
          {
            auto x = edgeQuad.localPoint(qp);
            auto y = intersection.geometryInInside().global(x);
            localFunction.evaluate( y, value );
            double weight = edgeQuad.weight(qp) * intersection.geometry().integrationElement(x) / intersection.geometry().volume();
            edgeBFS_.evaluateEach(x,
              [&](std::size_t alpha, typename LocalFunction::RangeType phi ) {
                int kk = alpha+k;
                //! SubDofWrapper has no size assert( kk < localDofVector.size() );
                localDofVector[ kk ] += value[0]*phi[0] * weight;
              }
            );
          }
        };
        auto inner = [&] (int poly,int i,int k,int numDofs)
        {
          assert(numDofs == innerShapeFunctionSet.size());
          //! SubVector has no size: assert(k+numDofs == localDofVector.size());
          InnerQuadratureType innerQuad( element, 2*polOrder_ );
          for (int qp=0;qp<innerQuad.nop();++qp)
          {
            auto y = innerQuad.point(qp);
            localFunction.evaluate( innerQuad[qp], value );
            double weight = innerQuad.weight(qp) * element.geometry().integrationElement(y) / indexSet_.volume(poly);
            innerShapeFunctionSet.evaluateEach(innerQuad[qp],
              [&](std::size_t alpha, typename LocalFunction::RangeType phi ) {
                int kk = alpha+k;
                //! SubVector has no size assert( kk < localDofVector.size() );
                localDofVector[ kk ] += value[0]*phi[0] * weight;
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
                                std::vector<int> &mask) const
      {
        const int dimension = AgglomerationIndexSet::dimension;
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
          if (indexSet_.testSpaces()[0]>=0)
          {
            for( ; i < refElement.size( edgeNumber, dimension-1, dimension ); ++i )
            {
              int vertexNumber = refElement.subEntity( edgeNumber, dimension-1, i, dimension);
              const int k = indexSet_.localIndex( element, vertexNumber, dimension );
              assert(k>=0); // intersection is 'outside' so vertices should be as well
              vertex(poly,vertexNumber,i,1);
              mask.push_back(k);
            }
          }
          if (indexSet_.testSpaces()[1]>=0)
          {
            edge(poly,intersection,i,edgeSize);
            for (int kk=0;kk<edgeSize;++kk)
              mask.push_back(k*edgeSize+edgeOffset+kk);
          }
        }
      }

      const AgglomerationIndexSet &indexSet_;
      const EdgeShapeFunctionSetType  edgeBFS_;
      const InnerShapeFunctionSetType innerBFS_;
      const unsigned int polOrder_;
    };



    // agglomerationVEMInterpolation
    // -----------------------------

    template< class AgglomerationIndexSet >
    inline AgglomerationVEMInterpolation< AgglomerationIndexSet >
    agglomerationVEMInterpolation ( const AgglomerationIndexSet &indexSet, unsigned int polOrder ) noexcept
    {
      return AgglomerationVEMInterpolation< AgglomerationIndexSet >( indexSet, polOrder );
    }

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_INTERPOLATION_HH
