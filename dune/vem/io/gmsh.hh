#ifndef DUNE_VEM_IO_GMSH_HH
#define DUNE_VEM_IO_GMSH_HH

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <utility>

#include <dune/common/iteratorrange.hh>
#include <dune/common/fvector.hh>
#include <dune/common/version.hh>

#include <dune/geometry/type.hh>
#include <dune/geometry/referenceelements.hh>

#include <dune/grid/common/gridfactory.hh>
#include <dune/grid/common/mcmgmapper.hh>
#include <dune/grid/common/rangegenerators.hh>

namespace Dune
{

  namespace Vem
  {

    namespace Gmsh
    {

      enum Format { ascii = 0, binary = 1 };

      struct ElementType
      {
        Dune::GeometryType duneType;
        std::size_t numNodes;
        std::unique_ptr< std::pair< unsigned int, unsigned int >[] > subEntity;
      };

      typedef std::vector< std::string > Section;
      typedef std::multimap< std::string, Section > SectionMap;

      struct Node
      {
        std::size_t id;
        FieldVector< double, 3 > position;
      };

      struct Element
      {
        std::size_t id = 0;
        const ElementType *type = nullptr;
        std::unique_ptr< std::size_t[] > nodes;
        std::size_t numTags = 0;
        std::unique_ptr< int[] > tags;
      };

      struct DuneEntity
      {
        DuneEntity ( std::size_t id, GeometryType type )
          : id_( id ), type_( type ),
            vertices_( std::make_unique< std::size_t[] >( size() ) )
        {}

        std::size_t id () const { return id_; }

        std::size_t size () const
        {
#if DUNE_VERSION_NEWER(DUNE_GEOMETRY, 2, 6)
          return Geo::Impl::size( type().id(), type().dim(), type().dim() );
#else // #if DUNE_VERSION_NEWER(DUNE_GEOMETRY, 2, 6)
          return Impl::size( type().id(), type().dim(), type().dim() );
#endif // #else // #if DUNE_VERSION_NEWER(DUNE_GEOMETRY, 2, 6)
        }

        GeometryType type () const { return type_; }

        IteratorRange< std::size_t * > vertices () { return { vertices_.get(), vertices_.get() + size() }; }
        IteratorRange< const std::size_t * > vertices () const { return { vertices_.get(), vertices_.get() + size() }; }

      private:
        std::size_t id_;
        GeometryType type_;
        std::unique_ptr< std::size_t[] > vertices_;
      };

      std::vector< DuneEntity > duneEntities ( const std::vector< Element > &elements, unsigned int dim );

      void findVertices ( const DuneEntity &entity, const std::vector< std::size_t > &vertices, std::vector< unsigned int > &indices );

      std::vector< Element > parseElements ( const SectionMap &sectionMap );
      std::vector< Node > parseNodes ( const SectionMap &sectionMap );
      std::tuple< double, Format, std::size_t > parseMeshFormat ( const SectionMap &sectionMap );

      SectionMap readFile ( const std::string &filename );

      std::vector< int > tags ( const std::vector< Element > &elements, const std::vector< std::size_t > &ids, std::size_t tag );

      std::vector< std::size_t > vertices ( const std::vector< DuneEntity > &entities );



      // Grid Factory Manipulation
      // -------------------------

      template< class Grid >
      inline void insertVertices ( GridFactory< Grid > &factory, const std::vector< std::size_t > &vertices, const std::vector< Node > &nodes )
      {
        const int dimWorld = Grid::dimensionworld;
        for( std::size_t v : vertices )
        {
          const auto pos = std::lower_bound( nodes.begin(), nodes.end(), v, [] ( const Node &a, std::size_t b ) { return (a.id < b); } );
          if( (pos == nodes.end()) || (pos->id != v) )
            DUNE_THROW( Exception, "Unable to find node " << v << " in nodes vector" );

          FieldVector< typename Grid::ctype, dimWorld > position( 0 );
          for( int i = 0; i < std::min( dimWorld, 3 ); ++i )
            position[ i ] = pos->position[ i ];

          factory.insertVertex( position );
        }
      }

      template< class Grid >
      inline void insertElements ( GridFactory< Grid > &factory, const std::vector< DuneEntity > &entities, const std::vector< std::size_t > &vertices )
      {
        std::vector< unsigned int > indices;
        for( const DuneEntity &entity : entities )
        {
          findVertices( entity, vertices, indices );
          factory.insertElement( entity.type(), indices );
        }
      }

      template< class GridView >
      inline std::vector< std::size_t > vertices ( const GridView &gridView, const GridFactory< typename GridView::Grid > &factory, const std::vector< std::size_t > &vertices )
      {
        // MultipleCodimMultipleGeomTypeMapper< GridView, MCMGVertexLayout > mapper( gridView );
        MultipleCodimMultipleGeomTypeMapper< GridView > mapper( gridView, mcmgVertexLayout() );
        std::vector< std::size_t > ids( mapper.size(), std::size_t( 0 ) );
        for( const auto vertex : vertices( gridView, Partitions::all ) )
          ids[ mapper.index( vertex ) ] = vertices[ factory.insertionIndex( vertex ) ];
        return std::move( ids );
      }

      template< class GridView >
      inline std::vector< std::size_t > elements ( const GridView &gridView, const GridFactory< typename GridView::Grid > &factory, const std::vector< DuneEntity > &entities )
      {
        // MultipleCodimMultipleGeomTypeMapper< GridView, MCMGElementLayout > mapper( gridView );
        MultipleCodimMultipleGeomTypeMapper< GridView > mapper( gridView, mcmgElementLayout() );
        std::vector< std::size_t > ids( mapper.size(), std::size_t( 0 ) );
        for( const auto element : elements( gridView, Partitions::all ) )
          ids[ mapper.index( element ) ] = entities[ factory.insertionIndex( element ) ].id();
        return std::move( ids );
      }

    } // namespace Gmsh

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_IO_GMSH_HH
