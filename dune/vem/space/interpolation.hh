#ifndef DUNE_VEM_SPACE_INTERPOLATION_HH
#define DUNE_VEM_SPACE_INTERPOLATION_HH

#include <cstddef>

#include <initializer_list>
#include <utility>

#include <dune/geometry/referenceelements.hh>

#include <dune/fem/space/shapefunctionset/orthonormal.hh>
#include <dune/fem/space/lagrange.hh>

#include <dune/vem/agglomeration/shapefunctionset.hh>
#include <dune/vem/agglomeration/dgspace.hh>

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

    private:
      typedef Dune::Fem::CachingQuadrature<GridPartType,0> InnerQuadratureType;
      typedef Dune::Fem::CachingQuadrature<GridPartType,1> EdgeQuadratureType;
      typedef typename ElementType::Geometry::ctype ctype;
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld-1,1> EdgeFSType;
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld,1> InnerFSType;
      typedef Dune::Fem::OrthonormalShapeFunctionSet<EdgeFSType> EdgeShapeFunctionSetType;
      typedef Dune::Fem::OrthonormalShapeFunctionSet<InnerFSType> InnerShapeFunctionSetType;

    public:
      explicit AgglomerationVEMInterpolation ( const AgglomerationIndexSet &indexSet ) noexcept
        : indexSet_( indexSet )
        , edgeSpace_( Dune::GeometryType(Dune::GeometryType::cube,GridPartType::dimension-1), std::max(polOrder-2,0) )
        , innerSpace_( Dune::GeometryType(Dune::GeometryType::cube,GridPartType::dimension-1), std::max(polOrder-2,0) )
      {}

      const GridPartType &gridPart() const { return indexSet_.agglomeration().gridPart(); }

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
      template< class LocalFunction, class LocalDofVector >
      void operator() ( const ElementType &element, const LocalFunction &localFunction, LocalDofVector &localDofVector ) const
      {
        typename LocalFunction::RangeType value;
        assert( value.dimension == 1 );
        const auto &bbox = indexSet_.boundingBox(element);
        BoundingBoxShapeFunctionSet< ElementType, InnerShapeFunctionSetType > innerShapeFunctionSet( element, bbox, innerSpace_ );

        auto vertex = [&] (int poly,auto i,int k,int numDofs)
        {
          const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );
          const auto &x = refElement.position( i, dimension );
          localFunction.evaluate( x, value );
          assert( k < localDofVector.size() );
          localDofVector[ k ] = value[ 0 ];
        };
        auto edge = [&] (int poly,auto intersection,int k,int numDofs)
        {
          const int el = indexSet_.index( intersection.inside() );
          bool twist = true;
          if (false && intersection.neighbor())
            twist = el < indexSet_.index( intersection.outside() );
          assert(numDofs == edgeSpace_.size());
          EdgeQuadratureType edgeQuad( gridPart(),
                intersection, 2*polOrder, EdgeQuadratureType::INSIDE );
          for (int qp=0;qp<edgeQuad.nop();++qp)
          {
            auto x = edgeQuad.localPoint(qp);
            if (!twist) x[0] = 1-x[0];
            auto y = intersection.geometryInInside().global(x);
            localFunction.evaluate( y, value );
            double weight = edgeQuad.weight(qp) * intersection.geometry().integrationElement(x) / intersection.geometry().volume();
            edgeSpace_.evaluateEach(x,
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
          InnerQuadratureType innerQuad( element, 2*polOrder );
          for (int qp=0;qp<innerQuad.nop();++qp)
          {
            auto y = innerQuad.point(qp);
            localFunction.evaluate( y, value );
            double weight = innerQuad.weight(qp) * element.geometry().integrationElement(y) / indexSet_.volume(poly);
            innerSpace_.evaluateEach(y,
              [&](std::size_t alpha, typename LocalFunction::RangeType phi ) {
                int kk = alpha+k;
                // std::cout << "inner kk=" << kk << " " << value[0] << " " << phi[0] << " " << weight
                //           << " " << element.geometry().integrationElement(y)
                //           << " " << indexSet_.volume(poly) << std::endl;
                assert( kk < localDofVector.size() );
                localDofVector[ kk ] += value[0]*phi[0] * weight;
              }
            );
          }
        };
        (*this)(element,vertex,edge,inner);
      }
      void operator() ( const ElementType &element, std::vector<bool> &mask) const
      {
        auto set = [&mask] (int poly,auto i,int k,int numDofs)
        { std::fill(mask.begin()+k,mask.begin()+k+numDofs,true); };
        (*this)(element,set,set,set);
      }
      template< class ShapeFunctionSet, class LocalDofMatrix >
      void operator() ( const BoundingBoxShapeFunctionSet< ElementType, ShapeFunctionSet > &shapeFunctionSet, LocalDofMatrix &localDofMatrix ) const
      {
        const ElementType &element = shapeFunctionSet.entity();
        const auto &bbox = indexSet_.boundingBox(element);
        BoundingBoxShapeFunctionSet< ElementType, InnerShapeFunctionSetType > innerShapeFunctionSet( element, bbox, innerSpace_ );
        auto vertex = [&] (int poly,int i,int k,int numDofs)
        {
          const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );
          const auto &x = refElement.position( i, dimension );
          shapeFunctionSet.evaluateEach( x, [ &localDofMatrix, k ] ( std::size_t alpha, typename ShapeFunctionSet::RangeType phi ) {
              assert( phi.dimension == 1 );
              localDofMatrix[ k ][ alpha ] = phi[ 0 ];
            } );
        };
        auto edge = [&] (int poly,auto intersection,int k,int numDofs)
        {
          const int el = indexSet_.index( intersection.inside() );
          bool twist = true;
          if (false && intersection.neighbor())
            twist = el < indexSet_.index( intersection.outside() );
          assert(numDofs == edgeSpace_.size());
          EdgeQuadratureType edgeQuad( gridPart(),
                intersection, 2*polOrder, EdgeQuadratureType::INSIDE );
          for (int qp=0;qp<edgeQuad.nop();++qp)
          {
            auto x = edgeQuad.localPoint(qp);
            if (!twist) x[0] = 1.-x[0];
            auto y = intersection.geometryInInside().global(x);
            double weight = edgeQuad.weight(qp) * intersection.geometry().integrationElement(x) / intersection.geometry().volume();
            shapeFunctionSet.evaluateEach( y,
              [ & ] ( std::size_t beta, typename ShapeFunctionSet::RangeType value )
              {
                edgeSpace_.evaluateEach( x,
                  [&](std::size_t alpha, typename EdgeFSType::RangeType phi ) {
                    int kk = alpha+k;
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
            shapeFunctionSet.evaluateEach( y,
              [ & ] ( std::size_t beta, typename ShapeFunctionSet::RangeType value )
              {
                innerShapeFunctionSet.evaluateEach( y,
                  [&](std::size_t alpha, typename InnerFSType::RangeType phi ) {
                    int kk = alpha+k;
                    localDofMatrix[ kk ][ beta ] += value[0]*phi[0] * weight;
                  }
                );
              }
            );
          }
        };
        (*this)(element,vertex,edge,inner);
      }

      static std::initializer_list< std::pair< int, unsigned int > > dofsPerCodim ()
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
      EdgeShapeFunctionSetType edgeSpace_;
      InnerShapeFunctionSetType innerSpace_;
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
