#ifndef DUNE_VEM_SPACE_INTERPOLATION_HH
#define DUNE_VEM_SPACE_INTERPOLATION_HH

#include <cstddef>

#include <initializer_list>
#include <utility>

#include <dune/geometry/referenceelements.hh>

#include <dune/fem/space/shapefunctionset/orthonormal.hh>

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
      static const int dimension = AgglomerationIndexSet::dimension;

      typedef typename AgglomerationIndexSet::ElementType ElementType;
      typedef typename AgglomerationIndexSet::GridPartType GridPartType;

    private:
      typedef typename ElementType::Geometry::ctype ctype;

    public:
      explicit AgglomerationVEMInterpolation ( const AgglomerationIndexSet &indexSet ) noexcept
        : indexSet_( indexSet )
        , edgeSpace_( Dune::GeometryType(Dune::GeometryType::cube,GridPartType::dimension-1), std::max(polOrder-22,0) )
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
                  << " numDofs: " << localDofVector.size()
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
            vertex(i,k,1);
#if 0
            std::cout << "    vertex = " << indexSet_.subIndex(element,i,dimension)
                      << " k=" << k
                      << " global=" << element.geometry().global(x)
                      << " value=" << value << std::endl;
#endif
          }
        }
        // edge dofs
        if (polOrder>1)
          for( int i = 0; i < refElement.size( dimension-1 ); ++i )
          {
            const int k = indexSet_.localIndex( element, i, dimension-1 )*edgeSize+edgeOffset;
            if( k >= 0 )
            {
              edge(i,k,edgeSize);
#if 0
              std::cout << "    edge = " << indexSet_.subIndex(element,i,dimension-1)
                        << "  k=" << kk << " - " << kk+edgeSize-1
                        << " global=" << element.geometry().global(x)
                        << " value=" << value << std::endl;
#endif
            }
          }
        // inner dofs
        if (polOrder>2)
        {
          assert(polOrder == 3);
          const int k = indexSet_.localIndex( element, 0, 0 ) + innerOffset;
          inner(0,k,innerSize);
#if 0
          std::cout << "    inner = " << indexSet_.subIndex(element,0,0)
                    << "  k=" << kk << " - " << kk+innerSize
                    << " global=" << element.geometry().global(x)
                    << " value=" << value << std::endl;
#endif
        }
      }
      template< class LocalFunction, class LocalDofVector >
      void operator() ( const ElementType &element, const LocalFunction &localFunction, LocalDofVector &localDofVector ) const
      {
        typename LocalFunction::RangeType value;
        assert( value.dimension == 1 );

        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );
        auto vertex = [&element,&localFunction,&localDofVector,&refElement,&value] (int i,int k,int numDofs)
        {
          const auto &x = refElement.position( i, dimension );
          localFunction.evaluate( x, value );
          assert( k < localDofVector.size() );
          localDofVector[ k ] = value[ 0 ];
        };
        auto edge = [&element,&localFunction,&localDofVector,&refElement,&value] (int i,int k,int numDofs)
        {
          double length = element.template subEntity<dimension-1>(i).geometry().volume();
          const auto &x = refElement.position( i, dimension-1 );
          localFunction.evaluate( x, value );
          for ( int kk = k; kk < k+numDofs; ++kk )
          {
            assert( kk < localDofVector.size() );
            localDofVector[ kk ] = value[ 0 ] * length;
          }
        };
        auto inner = [&element,&localFunction,&localDofVector,&refElement,&value] (int i,int k,int numDofs)
        {
          assert(polOrder == 3);
          double volume = element.geometry().volume();
          const auto &x = refElement.position( 0, 0 );
          localFunction.evaluate( x, value );
          for ( int kk = k; kk < k+numDofs; ++kk )
          {
            assert( kk < localDofVector.size() );
            localDofVector[ kk ] += value[ 0 ]*volume;
          }
        };
        (*this)(element,vertex,edge,inner);
      }
      void operator() ( const ElementType &element, std::vector<bool> &mask) const
      {
        std::fill(mask.begin(),mask.end(),false);
        auto set = [&mask] (int i,int k,int numDofs)
        { std::fill(mask.begin()+k,mask.begin()+k+numDofs,true); };
        (*this)(element,set,set,set);
      }
      template< class ShapeFunctionSet, class LocalDofMatrix >
      void operator() ( const BoundingBoxShapeFunctionSet< ElementType, ShapeFunctionSet > &shapeFunctionSet, LocalDofMatrix &localDofMatrix ) const
      {
        const ElementType &element = shapeFunctionSet.entity();
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );
        auto vertex = [&element,&shapeFunctionSet,&localDofMatrix,&refElement] (int i,int k,int numDofs)
        {
          const auto &x = refElement.position( i, dimension );
          shapeFunctionSet.evaluateEach( x, [ &localDofMatrix, k ] ( std::size_t alpha, typename ShapeFunctionSet::RangeType phi ) {
              assert( phi.dimension == 1 );
              localDofMatrix[ k ][ alpha ] = phi[ 0 ];
            } );
        };
        auto edge = [&element,&shapeFunctionSet,&localDofMatrix,&refElement] (int i,int k,int numDofs)
        {
          double length = element.template subEntity<dimension-1>(i).geometry().volume();
          const auto &x = refElement.position( i, dimension-1 );
          for ( int kk = k; kk < k+numDofs; ++kk )
            shapeFunctionSet.evaluateEach( x, [ &localDofMatrix, kk, length ] ( std::size_t alpha, typename ShapeFunctionSet::RangeType phi ) {
                localDofMatrix[ kk ][ alpha ] = phi[ 0 ]*length;
              } );
        };
        auto inner = [&element,&shapeFunctionSet,&localDofMatrix,&refElement] (int i,int k,int numDofs)
        {
          double volume = element.geometry().volume();
          const auto &x = refElement.position( 0, 0 );
          for ( int kk = k; kk < k+numDofs; ++kk )
            shapeFunctionSet.evaluateEach( x, [ &localDofMatrix, kk, volume ] ( std::size_t alpha, typename ShapeFunctionSet::RangeType phi ) {
                localDofMatrix[ kk ][ alpha ] += phi[ 0 ]*volume;
              } );
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
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld-1,1> EdgeFSType;
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld,1> InnerFSType;
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
