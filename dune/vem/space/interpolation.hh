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
#if 0
            std::cout << "    vertex = " << indexSet_.subIndex(element,i,dimension)
                      << " k=" << k;
#endif
            vertex(i,k,1);
          }
        }
        // edge dofs
        if (polOrder>1)
          for( int i = 0; i < refElement.size( dimension-1 ); ++i )
          {
            const int k = indexSet_.localIndex( element, i, dimension-1 )*edgeSize+edgeOffset;
            if( k >= edgeOffset )
            {
#if 0
              std::cout << "    edge = " << indexSet_.subIndex(element,i,dimension-1)
                        << "  k=" << k << " - " << k+edgeSize-1;
#endif
              edge(i,k,edgeSize);
            }
          }
        // inner dofs
        if (polOrder>2)
        {
          assert(polOrder == 3);
          const int k = indexSet_.localIndex( element, 0, 0 ) + innerOffset;
#if 0
          std::cout << "    inner = " << indexSet_.subIndex(element,0,0)
                    << "  k=" << k << " - " << k+innerSize;
#endif
          inner(0,k,innerSize);
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
          // std::cout << " ... " << k;
          const auto &x = refElement.position( i, dimension );
          localFunction.evaluate( x, value );
          assert( k < localDofVector.size() );
          localDofVector[ k ] = value[ 0 ];
          // std::cout << std::endl;
        };
        int lagSize = edgeSpace_.blockMapper().numDofs(element);
        std::vector< bool> globalBlockDofsFilter(lagSize);
        std::vector<double> lagDofs(lagSize);
        edgeSpace.interpolate(element)(localFunction,lagDofs);
        auto edge = [&element,&localFunction,&localDofVector,&refElement,&value,&lagDofs] (int i,int k,int numDofs)
        {
          space_.blockMapper().onSubEntity(element,i,1,globalBlockDofsFilter);
          // double length = element.template subEntity<dimension-1>(i).geometry().volume();
          // const auto &x = refElement.position( i, dimension-1 );
          // localFunction.evaluate( x, value );
          auto itFilter = globalBlockDofsFilter.begin();
          auto itValues = lagDofs.begin();
          for ( int kk = k; kk < k+numDofs; ++kk )
          {
            for ( ; *itFilter; ++itFilter,++itValues );
            // std::cout << " ... " << kk;
            assert( kk < localDofVector.size() );
            // localDofVector[ kk ] = value[ 0 ] * length;
            localDofVector[ kk ] = *itValues;
          }
          // std::cout << std::endl;
        };
        auto inner = [&element,&localFunction,&localDofVector,&refElement,&value] (int i,int k,int numDofs)
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
          // std::cout << " ... " << k;
          const auto &x = refElement.position( i, dimension );
          shapeFunctionSet.evaluateEach( x, [ &localDofMatrix, k ] ( std::size_t alpha, typename ShapeFunctionSet::RangeType phi ) {
              assert( phi.dimension == 1 );
              localDofMatrix[ k ][ alpha ] = phi[ 0 ];
            } );
          // std::cout << std::endl;
        };
        auto edge = [&element,&shapeFunctionSet,&localDofMatrix,&refElement] (int i,int k,int numDofs)
        {
          double length = element.template subEntity<dimension-1>(i).geometry().volume();
          // const auto &x = refElement.position( i, dimension-1 );
          for ( int kk = k; kk < k+numDofs; ++kk )
          {
            // std::cout << " ... " << kk;
            Dune::FieldVector<double,1> x({double(kk-k+1)/double(numDofs+1)});
            shapeFunctionSet.evaluateEach( x, [ &localDofMatrix, kk, length ] ( std::size_t alpha, typename ShapeFunctionSet::RangeType phi ) {
                localDofMatrix[ kk ][ alpha ] = phi[ 0 ]*length;
              } );
          }
          // std::cout << std::endl;
        };
        auto inner = [&element,&shapeFunctionSet,&localDofMatrix,&refElement] (int i,int k,int numDofs)
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
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld-1,1> EdgeFSType;
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld,1> InnerFSType;
      const AgglomerationIndexSet &indexSet_;
      Dune::Fem::LagrangeSpace<FSType> edgeSpace_;
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
