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
      typedef Dune::Fem::CachingQuadrature<GridPartType,1> EdgeQuadratureType;
      typedef typename ElementType::Geometry::ctype ctype;
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld-1,1> EdgeFSType;
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld,1> InnerFSType;

    public:
      explicit AgglomerationVEMInterpolation ( const AgglomerationIndexSet &indexSet ) noexcept
        : indexSet_( indexSet )
        , edgeSpace_( Dune::GeometryType(Dune::GeometryType::cube,GridPartType::dimension-1), std::max(polOrder-2,0) )
        , innerSpace_( indexSet.agglomeration() )
      {}

      template< class Vertex, class Edge, class Inner>
      void operator() ( const ElementType &element,
          const Vertex &vertex, const Edge &edge, const Inner &inner) const
      {
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );

        const int poly = indexSet_.index( element );
        const int edgeOffset = indexSet_.subAgglomerates(poly,dimension);
        const int edgeSize = Dune::Fem::OrthonormalShapeFunctions< GridPartType::dimension-1 >::size( std::max(polOrder-2,0) );
        const int innerOffset = edgeOffset*edgeSize + indexSet_.subAgglomerates(poly,dimension-1);
        const int innerSize = Dune::Fem::OrthonormalShapeFunctions< GridPartType::dimension >::size( std::max(polOrder-3,0) );
#if 0
        std::cout << "polygon: " << poly
                  << " edgeOffset " << edgeOffset
                  << " innerOffset " << innerOffset
                  << std::endl;
#endif
        // vertex dofs
        for( int i = 0; i < refElement.size( dimension ); ++i )
        {
          const int k = indexSet_.localIndex( element, i, dimension );
          if( k >= 0 )
          {
            // std::cout << "    vertex = " << indexSet_.subIndex(element,i,dimension) << " k=" << k;
            vertex(i,k,1);
          }
        }
        // edge dofs
        if (polOrder>1)
        {
          // to avoid any issue with twists we use an intersection iterator
          // here instead of going over the edges
          auto it = innerSpace_.gridPart().ibegin( element );
          const auto endit = innerSpace_.gridPart().iend( element );
          for( ; it != endit; ++it )
          {
            const auto& intersection = *it;
            const int i = intersection.indexInInside();
            const int k = indexSet_.localIndex( element, i, dimension-1 )*edgeSize+edgeOffset;
            if (k>=edgeOffset)
            {
              // std::cout << "    edge = " << indexSet_.subIndex(element,i,dimension-1) << "  k=" << k << " - " << k+edgeSize-1;
              edge(intersection,k,edgeSize);
            }
          }
        }
        // inner dofs
        if (polOrder>2)
        {
          assert(polOrder == 3);
          const int k = indexSet_.localIndex( element, 0, 0 ) + innerOffset;
          // std::cout << "    inner = " << indexSet_.subIndex(element,0,0) << "  k=" << k << " - " << k+innerSize;
          inner(0,k,innerSize);
        }
      }
      template< class LocalFunction, class LocalDofVector >
      void operator() ( const ElementType &element, const LocalFunction &localFunction, LocalDofVector &localDofVector ) const
      {
        // std::fill(localDofVector.begin(), localDofVector.end(),0);
        typename LocalFunction::RangeType value;
        assert( value.dimension == 1 );

        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );
        auto vertex = [&] (auto i,int k,int numDofs)
        {
          // std::cout << " ... " << k;
          const auto &x = refElement.position( i, dimension );
          localFunction.evaluate( x, value );
          assert( k < localDofVector.size() );
          localDofVector[ k ] = value[ 0 ];
          // std::cout << std::endl;
        };
        auto edge = [&] (auto intersection,int k,int numDofs)
        {
          const int el = indexSet_.index( intersection.inside() );
          bool twist = true;
          if (false && intersection.neighbor())
            twist = el < indexSet_.index( intersection.outside() );
          assert(numDofs == edgeSpace_.size());
          EdgeQuadratureType edgeQuad( innerSpace_.gridPart(),
                intersection, 2*polOrder, EdgeQuadratureType::INSIDE );
#if 0
          std::cout << "in edge for LF (" << k << ") twist=" << twist
                    << " " << intersection.geometry().center()
                    << ": " << std::endl;
#endif
          for (int qp=0;qp<edgeQuad.nop();++qp)
          {
            auto x = edgeQuad.localPoint(qp);
            if (!twist) x[0] = 1-x[0];
            auto y = intersection.geometryInInside().global(x);
            localFunction.evaluate( y, value );
            edgeSpace_.evaluateEach(x,
              [&](std::size_t alpha, typename LocalFunction::RangeType phi ) {
                int kk = alpha+k;
#if 0
                std::cout << " ... (" << kk << "," << qp << ") ";
                std::cout << x
                          << " " << element.geometry().global(y)
                          << " " << value << " " << phi << std::endl;
#endif
                assert( kk < localDofVector.size() );
                localDofVector[ kk ] += value[0]*phi[0]
                     * edgeQuad.weight(qp); // *intersection.geometry().integrationElement(x);
              }
            );
          }
          // std::cout << std::endl;
        };
        auto inner = [&] (int i,int k,int numDofs)
        {
          assert(polOrder == 3);
          double volume = element.geometry().volume();
          const auto &x = refElement.position( 0, 0 );
          localFunction.evaluate( x, value );
          for ( int kk = k; kk < k+numDofs; ++kk )
          {
            // std::cout << " ... " << kk;
            assert( kk < localDofVector.size() );
            localDofVector[ kk ] += value[ 0 ]*volume;
          }
          // std::cout << std::endl;
        };
        (*this)(element,vertex,edge,inner);
      }
      void operator() ( const ElementType &element, std::vector<bool> &mask) const
      {
        // std::cout << std::endl;
        std::fill(mask.begin(),mask.end(),false);
        auto set = [&mask] (auto i,int k,int numDofs)
        { std::fill(mask.begin()+k,mask.begin()+k+numDofs,true); };
        (*this)(element,set,set,set);
      }
      template< class ShapeFunctionSet, class LocalDofMatrix >
      void operator() ( const BoundingBoxShapeFunctionSet< ElementType, ShapeFunctionSet > &shapeFunctionSet, LocalDofMatrix &localDofMatrix ) const
      {
        const ElementType &element = shapeFunctionSet.entity();
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );
        auto vertex = [&] (int i,int k,int numDofs)
        {
          // std::cout << " ... " << k;
          const auto &x = refElement.position( i, dimension );
          shapeFunctionSet.evaluateEach( x, [ &localDofMatrix, k ] ( std::size_t alpha, typename ShapeFunctionSet::RangeType phi ) {
              assert( phi.dimension == 1 );
              localDofMatrix[ k ][ alpha ] = phi[ 0 ];
            } );
          // std::cout << std::endl;
        };
        auto edge = [&] (auto intersection,int k,int numDofs)
        {
          // std::cout << "edge for SFS(" << k << "): ";
          const int el = indexSet_.index( intersection.inside() );
          bool twist = true;
          if (false && intersection.neighbor())
            twist = el < indexSet_.index( intersection.outside() );
          assert(numDofs == edgeSpace_.size());
          EdgeQuadratureType edgeQuad( innerSpace_.gridPart(),
                intersection, 2*polOrder, EdgeQuadratureType::INSIDE );
          for (int qp=0;qp<edgeQuad.nop();++qp)
          {
            auto x = edgeQuad.localPoint(qp);
            if (!twist) x[0] = 1.-x[0];
            auto y = intersection.geometryInInside().global(x);
            shapeFunctionSet.evaluateEach( y,
              [ & ] ( std::size_t beta, typename ShapeFunctionSet::RangeType value )
              {
                edgeSpace_.evaluateEach( x,
                  [&](std::size_t alpha, typename EdgeFSType::RangeType phi ) {
                    int kk = alpha+k;
                    // std::cout << " ... (" << kk << "," << qp << ") ";
                    localDofMatrix[ kk ][ beta ] += value[0]*phi[0]
                        * edgeQuad.weight(qp); // *intersection.geometry().integrationElement(x);
                  }
                );
              }
            );
          }
          // std::cout << std::endl;
        };

        auto inner = [&] (int i,int k,int numDofs)
        {
          double volume = element.geometry().volume();
          const auto &x = refElement.position( 0, 0 );
          for ( int kk = k; kk < k+numDofs; ++kk )
          {
            // std::cout << " ... " << kk;
            shapeFunctionSet.evaluateEach( x, [ &localDofMatrix, kk, volume ] ( std::size_t alpha, typename ShapeFunctionSet::RangeType phi ) {
                localDofMatrix[ kk ][ alpha ] += phi[ 0 ]*volume;
              } );
          }
          // std::cout << std::endl;
        };
        (*this)(element,vertex,edge,inner);
      }

      static std::initializer_list< std::pair< int, unsigned int > > dofsPerCodim ()
      {
        const int eSize = Dune::Fem::OrthonormalShapeFunctions< GridPartType::dimension-1 >::size( std::max(polOrder-2,0) );
        const int iSize = Dune::Fem::OrthonormalShapeFunctions< GridPartType::dimension >::size( std::max(polOrder-3,0) );
        switch (polOrder)
        {
        case 1:
          return { std::make_pair( dimension, 1u ) };
        case 2:
          return { std::make_pair( dimension, 1u ),
                   std::make_pair( dimension-1, eSize ) };
        default:
          return { std::make_pair( dimension, 1u ),
                   std::make_pair( dimension-1, eSize ),
                   std::make_pair( dimension-2, iSize ) };
        }
      }

    private:
      const AgglomerationIndexSet &indexSet_;
      Dune::Fem::OrthonormalShapeFunctionSet<EdgeFSType> edgeSpace_;
      AgglomerationDGSpace<InnerFSType,GridPartType,std::max(polOrder-3,0)> innerSpace_;
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
