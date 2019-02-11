#ifndef DUNE_VEM_SPACE_INTERPOLATION_HH
#define DUNE_VEM_SPACE_INTERPOLATION_HH

#include <cstddef>

#include <initializer_list>
#include <utility>

#include <dune/common/dynmatrix.hh>
#include <dune/geometry/referenceelements.hh>

#include <dune/fem/quadrature/elementquadrature.hh>
#include <dune/fem/space/shapefunctionset/orthonormal.hh>
#include <dune/vem/agglomeration/shapefunctionset.hh>

namespace Dune
{

  namespace Vem
  {

    // AgglomerationVEMInterpolation
    // -----------------------------

    template< class AgglomerationIndexSet, int polOrder >
    class AgglomerationVEMInterpolation
    {
      typedef AgglomerationVEMInterpolation< AgglomerationIndexSet, polOrder > ThisType;

    public:
      typedef typename AgglomerationIndexSet::ElementType ElementType;
      typedef typename AgglomerationIndexSet::GridPartType GridPartType;
      static const int dimension = AgglomerationIndexSet::dimension;
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
      explicit AgglomerationVEMInterpolation ( const AgglomerationIndexSet &indexSet ) noexcept
        : indexSet_( indexSet )
        , edgeBFS_( Dune::GeometryType(Dune::GeometryType::cube,GridPartType::dimension-1), std::max(polOrder-2,0) )
        , innerBFS_( Dune::GeometryType(Dune::GeometryType::cube,GridPartType::dimension), std::max(polOrder-2,0) )
      {}

      const GridPartType &gridPart() const { return indexSet_.agglomeration().gridPart(); }

      // carry out actual interpolation giving the three components, i.e.,
      // for the vertex, edge, and inner parts.
      // This calls these interpolation operators with the correct indices
      // to fill the dof vector or the matrix components
      template< class Vertex, class Edge, class Inner>
      void operator() ( const ElementType &element,
          const Vertex &vertex, const Edge &edge, const Inner &inner) const
      {
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );

        const int poly = indexSet_.index( element );
        const int edgeOffset = indexSet_.subAgglomerates(poly,dimension);
        const int edgeSize = Dune::Fem::OrthonormalShapeFunctions< GridPartType::dimension-1 >::size( std::max(polOrder-2,0) );
        const int innerOffset = edgeOffset*edgeSize + indexSet_.subAgglomerates(poly,dimension-1);
        const int innerSize = Dune::Fem::OrthonormalShapeFunctions< GridPartType::dimension >::size( std::max(polOrder-2,0) );
        // vertex dofs
        for( int i = 0; i < refElement.size( dimension ); ++i )
        {
          const int k = indexSet_.localIndex( element, i, dimension );
          if( k >= 0 )
            vertex(poly,i,k,1);
        }
        // edge dofs
        if (polOrder>1)
        {
          // to avoid any issue with twists we use an intersection iterator
          // here instead of going over the edges
          auto it = gridPart().ibegin( element );
          const auto endit = gridPart().iend( element );
          for( ; it != endit; ++it )
          {
            const auto& intersection = *it;
            const int i = intersection.indexInInside();
            const int k = indexSet_.localIndex( element, i, dimension-1 )*edgeSize+edgeOffset;
            if (k>=edgeOffset)
              edge(poly,intersection,k,edgeSize);
          }
          // inner dofs
          const int k = indexSet_.localIndex( element, 0, 0 ) + innerOffset;
          inner(poly,0,k,innerSize);
        }
      }

      // perform interpolation for a single localized function
      template< class LocalFunction, class LocalDofVector >
      void operator() ( const ElementType &element, const LocalFunction &localFunction,
                        LocalDofVector &localDofVector ) const
      {
        typename LocalFunction::RangeType value;
        assert( value.dimension == 1 );
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );
        const auto &bbox = indexSet_.boundingBox(element);
        BoundingBoxShapeFunctionSet< ElementType, InnerShapeFunctionSetType > innerShapeFunctionSet( element, bbox, innerBFS_ );

        // define the vertex,edge, and inner parts of the interpolation
        auto vertex = [&] (int poly,auto i,int k,int numDofs)
        {
          const auto &x = refElement.position( i, dimension );
          localFunction.evaluate( x, value );
          assert( k < localDofVector.size() );
          localDofVector[ k ] = value[ 0 ];
        };
        auto edge = [&] (int poly,auto intersection,int k,int numDofs)
        {
          assert(numDofs == edgeBFS_.size());
          int edgeNumber = intersection.indexInInside();
#if 0
          const auto &idSet = gridPart().grid().localIdSet();
          const auto left = idSet.subId( element, refElement.subEntity( edgeNumber, dimension-1, 0, dimension ), dimension );
          const auto right = idSet.subId( element, refElement.subEntity( edgeNumber, dimension-1, 1, dimension ), dimension );
          bool noTwist = true; // left < right;
#endif
          EdgeQuadratureType edgeQuad( gridPart(),
                intersection, 2*polOrder, EdgeQuadratureType::INSIDE );
          for (int qp=0;qp<edgeQuad.nop();++qp)
          {
            auto x = edgeQuad.localPoint(qp);
            auto y = intersection.geometryInInside().global(x);
            localFunction.evaluate( y, value );
            double weight = edgeQuad.weight(qp) * intersection.geometry().integrationElement(x) / intersection.geometry().volume();
            edgeBFS_.evaluateEach(x,
              [&](std::size_t alpha, typename LocalFunction::RangeType phi ) {
                int kk = alpha+k;
                assert( kk < localDofVector.size() );
                localDofVector[ kk ] += value[0]*phi[0] * weight;
              }
            );
          }
        };
        auto inner = [&] (int poly,int i,int k,int numDofs)
        {
          assert(numDofs == innerShapeFunctionSet.size());
          assert(k+numDofs == localDofVector.size());
          InnerQuadratureType innerQuad( element, 2*polOrder );
          for (int qp=0;qp<innerQuad.nop();++qp)
          {
            auto y = innerQuad.point(qp);
            localFunction.evaluate( innerQuad[qp], value );
            double weight = innerQuad.weight(qp) * element.geometry().integrationElement(y) / indexSet_.volume(poly);
            innerShapeFunctionSet.evaluateEach(innerQuad[qp],
              [&](std::size_t alpha, typename LocalFunction::RangeType phi ) {
                int kk = alpha+k;
                assert( kk < localDofVector.size() );
                localDofVector[ kk ] += value[0]*phi[0] * weight;
              }
            );
          }
        };
        (*this)(element,vertex,edge,inner);
      }

      // fill a mask vector providing the information which dofs are
      // 'active' on the given element
      void operator() ( const ElementType &element, std::vector<bool> &mask) const
      {
        auto set = [&mask] (int poly,auto i,int k,int numDofs)
        { std::fill(mask.begin()+k,mask.begin()+k+numDofs,true); };
        (*this)(element,set,set,set);
      }
      // preform interpolation of a full shape function set filling a transformation matrix
      template< class ShapeFunctionSet, class LocalDofMatrix >
      void operator() ( const BoundingBoxShapeFunctionSet< ElementType, ShapeFunctionSet > &shapeFunctionSet,
                       LocalDofMatrix &localDofMatrix ) const
      {
        const ElementType &element = shapeFunctionSet.entity();
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );
        const auto &bbox = indexSet_.boundingBox(element);
        BoundingBoxShapeFunctionSet< ElementType, InnerShapeFunctionSetType > innerShapeFunctionSet( element, bbox, innerBFS_ );

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
#if 0
          const auto &idSet = gridPart().grid().localIdSet();
          const auto left = idSet.subId( element, refElement.subEntity( edgeNumber, dimension-1, 0, dimension ), dimension );
          const auto right = idSet.subId( element, refElement.subEntity( edgeNumber, dimension-1, 1, dimension ), dimension );
          bool noTwist = true; // left < right;
#endif
          EdgeQuadratureType edgeQuad( gridPart(), intersection, 2*polOrder, EdgeQuadratureType::INSIDE );
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
          InnerQuadratureType innerQuad( element, 2*polOrder );
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
        (*this)(element,vertex,edge,inner);
      }

      ///////////////////////////////////////////////////////////////////////////
      // interpolation onto a single intersection
      // (bool argument needed to distinguish from the method following this one)
      template< class Vertex, class Edge>
      void operator() ( const IntersectionType &intersection,
                        const Vertex &vertex, const Edge &edge,
                        std::vector<int> &mask, bool tmp) const
      {
        const ElementType &element = intersection.inside();
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );

        const int poly = indexSet_.index( element );
        const int edgeOffset = indexSet_.subAgglomerates(poly,dimension);
        const int edgeSize = Dune::Fem::OrthonormalShapeFunctions< GridPartType::dimension-1 >::size( std::max(polOrder-2,0) );
        int edgeNumber = intersection.indexInInside();
        const int k = indexSet_.localIndex( element, edgeNumber, dimension-1 );
        assert(k>=0);
#if 0
        const auto &idSet = gridPart().grid().localIdSet();
        const auto left = idSet.subId( element, refElement.subEntity( edgeNumber, dimension-1, 0, dimension ), dimension );
        const auto right = idSet.subId( element, refElement.subEntity( edgeNumber, dimension-1, 1, dimension ), dimension );
        bool noTwist = true; // left < right;
#endif
        if (k>=0)
        {
          std::size_t i = 0;
          for( ; i < refElement.size( edgeNumber, dimension-1, dimension ); ++i )
          {
            int vertexNumber = refElement.subEntity( edgeNumber, dimension-1, i, dimension);
            const int k = indexSet_.localIndex( element, vertexNumber, dimension );
            assert(k>=0);
            vertex(poly,vertexNumber,i,1);
            mask.push_back(k);
          }
          if (polOrder>1)
          {
            edge(poly,intersection,i,edgeSize);
            for (int kk=0;kk<edgeSize;++kk)
              mask.push_back(k*edgeSize+edgeOffset+kk);
          }
        }
      }

      // interpolate the full shape function set on intersection needed for
      // the gradient projection matrix
      template< class ShapeFunctionSet >
      void operator() (const IntersectionType &intersection,
                       const ShapeFunctionSet &shapeFunctionSet, Dune::DynamicMatrix<double> &localDofMatrix,
                       std::vector<int> &mask) const
      {
        mask.clear();
        const ElementType &element = intersection.inside();
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );
        int edgeNumber = intersection.indexInInside();
        const auto &edgeGeo = refElement.template geometry<1>(edgeNumber);

#if 0
        const auto &idSet = gridPart().grid().localIdSet();
        const auto left = idSet.subId( element, refElement.subEntity( edgeNumber, dimension-1, 0, dimension ), dimension );
        const auto right = idSet.subId( element, refElement.subEntity( edgeNumber, dimension-1, 1, dimension ), dimension );
        bool noTwist = true; // left < right;
#endif

        // define the three relevant part of the interpolation, i.e.,
        // vertices,edges - no inner needed since only doing interpolation
        // on intersection
        auto vertex = [&] (int poly,int i,int k,int numDofs)
        {
          const auto &x = edgeGeo.local( refElement.position( i, dimension ) );
          // auto y = edgeGeo.global(x);
          // y -= refElement.position( i, dimension );
          // assert( y.two_norm() < 1e-10 );

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
                intersection, 2*polOrder, EdgeQuadratureType::INSIDE );
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
        (*this)(intersection,vertex,edge,mask,false);
      }

      // return the number of dofs per codimension
      static std::vector< std::pair< int, unsigned int > > dofsPerCodim ()
      {
        const int eSize = Dune::Fem::OrthonormalShapeFunctions< GridPartType::dimension-1 >::size( std::max(polOrder-2,0) );
        const int iSize = Dune::Fem::OrthonormalShapeFunctions< GridPartType::dimension >::size( std::max(polOrder-2,0) );
        switch (polOrder)
        {
        case 1:
          return { std::make_pair( dimension, 1u ) };
        default:
          return { std::make_pair( dimension, 1u ),
                   std::make_pair( dimension-1, eSize ),
                   std::make_pair( dimension-2, iSize ) };
        }
      }

    private:
      const AgglomerationIndexSet &indexSet_;
      EdgeShapeFunctionSetType edgeBFS_;
      InnerShapeFunctionSetType innerBFS_;
    };



    // agglomerationVEMInterpolation
    // -----------------------------

    template< int polOrder, class AgglomerationIndexSet >
    inline static AgglomerationVEMInterpolation< AgglomerationIndexSet, polOrder >
    agglomerationVEMInterpolation ( const AgglomerationIndexSet &indexSet ) noexcept
    {
      return AgglomerationVEMInterpolation< AgglomerationIndexSet, polOrder >( indexSet );
    }

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_INTERPOLATION_HH
