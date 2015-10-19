#include <config.h>

#include <cassert>

#include <algorithm>
#include <fstream>
#include <initializer_list>
#include <sstream>

#include <dune/common/exceptions.hh>

#include <dune/vem/io/gmsh.hh>

namespace Dune
{

  namespace Vem
  {

    namespace Gmsh
    {

      // ElementTypeImpl
      // ---------------

      struct ElementTypeImpl
        : public ElementType
      {
        ElementTypeImpl ( std::size_t id, Dune::GeometryType::BasicType basicType, int dim,
                          std::initializer_list< std::pair< unsigned int, unsigned int > > subs )
        {
          identifier = id;
          duneType = GeometryType( basicType, dim );
          numNodes = subs.size();
          subEntity = Std::make_unique< std::pair< unsigned int, unsigned int >[] >( numNodes );
          std::copy( subs.begin(), subs.end(), subEntity.get() );
        }

        operator std::pair< const std::size_t, const ElementType * > () const { return std::make_pair( identifier, this ); }

        std::size_t identifier;
      };



      // Element Types
      // -------------

      static const ElementTypeImpl order1Line( 1, GeometryType::cube, 1, { { 0, 1 }, { 1, 1 } } );
      static const ElementTypeImpl order1Triangle( 2, GeometryType::simplex, 2, { { 0, 2 }, { 1, 2 }, { 2, 2 } } );
      static const ElementTypeImpl order1Quadrangle( 3, GeometryType::cube, 2, { { 0, 2 }, { 1, 2 }, { 3, 2 }, { 2, 2 } } );
      static const ElementTypeImpl order1Tetrahedron( 4, GeometryType::simplex, 3, { { 0, 3 }, { 1, 3 }, { 2, 3 }, { 3, 3 } } );
      static const ElementTypeImpl order1Hexahedron( 5, GeometryType::cube, 3, { { 0, 3 }, { 1, 3 }, { 3, 3 }, { 2, 3 }, { 4, 3 }, { 5, 3 }, { 7, 3 }, { 6, 3 } } );
      static const ElementTypeImpl order1Prism( 6, GeometryType::prism, 3, { { 0, 3 }, { 1, 3 }, { 2, 3 }, { 3, 3 }, { 4, 3 }, { 5, 3 } } );
      static const ElementTypeImpl order1Pyramid( 7, GeometryType::pyramid, 3 , { { 0, 3 }, { 1, 3 }, { 3, 3 }, { 2, 3 }, { 4, 3 } } );

      static const ElementTypeImpl order2Line( 8, GeometryType::cube, 1, { { 0, 1 }, { 1, 1 }, { 0, 0 } } );
      static const ElementTypeImpl order2Triangle( 9, GeometryType::simplex, 2, { { 0, 2 }, { 1, 2 }, { 2, 2 }, { 0, 1 }, { 2, 1 }, { 1, 1 } } );
      static const ElementTypeImpl order2Quadrangle( 10, GeometryType::cube, 2, { { 0, 2 }, { 1, 2 }, { 3, 2 }, { 2, 2 }, { 2, 1 }, { 1, 1 }, { 3, 1 }, { 0, 1 }, { 0, 0 } } );

      static const ElementTypeImpl point( 15, GeometryType::cube, 0, { { 0, 0 } } );

      static const ElementTypeImpl reducedOrder2Quadrangle( 16, GeometryType::cube, 2, { { 0, 2 }, { 1, 2 }, { 3, 2 }, { 2, 2 }, { 2, 1 }, { 1, 1 }, { 3, 1 }, { 0, 1 } } );



      // makeElementTypes
      // ----------------

      static std::map< std::size_t, const ElementType * > makeElementTypes ()
      {
        std::map< std::size_t, const ElementType * > types;
        types.insert( order1Line );
        types.insert( order1Triangle );
        types.insert( order1Quadrangle );
        types.insert( order1Tetrahedron );
        types.insert( order1Prism );
        types.insert( order1Pyramid );
        types.insert( order2Line );
        types.insert( order2Triangle );
        types.insert( order2Quadrangle );
        types.insert( point );
        types.insert( reducedOrder2Quadrangle );
        return std::move( types );
      }



      // duneEntities
      // ------------

      std::vector< DuneEntity > duneEntities ( const std::vector< Element > &elements, unsigned int dim )
      {
        std::vector< DuneEntity > entities;
        for( const Element &element : elements )
        {
          if( element.type->duneType.dim() != dim )
            continue;

          DuneEntity entity( element.id, element.type->duneType );;
          for( std::size_t i = 0; i < element.type->numNodes; ++i )
          {
            if( element.type->subEntity[ i ].second == dim )
              entity.vertices().begin()[ element.type->subEntity[ i ].first ] = element.nodes[ i ];
          }

          entities.push_back( std::move( entity ) );
        }
        return std::move( entities );
      }



      // findUniqueSection
      // -----------------

      inline static SectionMap::const_iterator findUniqueSection ( const SectionMap &sectionMap, const std::string &sectionName )
      {
        if( sectionMap.count( sectionName ) != std::size_t( 1 ) )
          DUNE_THROW( IOError, "A Gmsh file requires exactly one '" + sectionName + "' section" );
        SectionMap::const_iterator section = sectionMap.find( sectionName );
        assert( section != sectionMap.end() );
        return section;
      }



      // findVertices
      // ------------

      void findVertices ( const DuneEntity &entity, const std::vector< std::size_t > &vertices, std::vector< unsigned int > &indices )
      {
        indices.resize( entity.size() );
        std::transform( entity.vertices().begin(), entity.vertices().end(), indices.begin(), [ &vertices ] ( std::size_t id ) {
            const auto pos = std::lower_bound( vertices.begin(), vertices.end(), id );
            if( (pos == vertices.end()) || (*pos != id) )
              DUNE_THROW( RangeError, "No such vertex: " << id );
            return (pos - vertices.begin());
          } );
      }



      // parseElements
      // -------------

      std::vector< Element > parseElements ( const SectionMap &sectionMap )
      {
        SectionMap::const_iterator section = findUniqueSection( sectionMap, "Elements" );
        if( section->second.empty() )
          DUNE_THROW( IOError, "Section 'Elements' must contain at least one line" );

        std::istringstream input( section->second.front() );
        std::size_t numElements = 0;
        input >> numElements;
        if( !input || (section->second.size() != numElements+1) )
          DUNE_THROW( IOError, "Section 'Elements' must contain exactly numElements+1 lines." );

        const std::map< std::size_t, const ElementType * > types = makeElementTypes();

        std::vector< Element > elements( numElements );
        for( std::size_t i = 0; i < numElements; ++i )
        {
          std::istringstream input( section->second[ i+1 ] );
          std::size_t typeId = 0;
          input >> elements[ i ].id >> typeId >> elements[ i ].numTags;
          if( !input )
            DUNE_THROW( IOError, "Unable to read line " << (i+1) << " of 'Elements' section" );

          const auto typeIt = types.find( typeId );
          if( typeIt == types.end() )
            DUNE_THROW( IOError, "Unknown element type " << typeId << " encountered in 'Elements' section" );
          elements[ i ].type = typeIt->second;

          if( elements[ i ].numTags > 4096 )
            DUNE_THROW( IOError, "Too many element tags encountered in 'Elements' section" );
          elements[ i ].tags = Std::make_unique< int[] >( elements[ i ].numTags );
          for( std::size_t j = 0; j < elements[ i ].numTags; ++j )
            input >> elements[ i ].tags[ j ];

          elements[ i ].nodes = Std::make_unique< std::size_t[] >( elements[ i ].type->numNodes );
          for( std::size_t j = 0; j < elements[ i ].type->numNodes; ++j )
            input >> elements[ i ].nodes[ j ];

          if( !input )
            DUNE_THROW( IOError, "Unable to read line " << (i+1) << " of 'Elements' section" );
        }

        // sort elements and ensure there are no duplicates
        std::sort( elements.begin(), elements.end(), [] ( const Element &a, const Element &b ) { return (a.id < b.id); } );
        const auto pos = std::adjacent_find( elements.begin(), elements.end(), [] ( const Element &a, const Element &b ) { return (a.id == b.id); } );
        if( pos != elements.end() )
          DUNE_THROW( IOError, "Duplicate element " << pos->id << " in 'Elements' section" );

        return std::move( elements );
      }



      // parseNodes
      // ----------

      std::vector< Node > parseNodes ( const SectionMap &sectionMap )
      {
        SectionMap::const_iterator section = findUniqueSection( sectionMap, "Nodes" );
        if( section->second.empty() )
          DUNE_THROW( IOError, "Section 'Nodes' must contain at least one line" );

        std::istringstream input( section->second.front() );
        std::size_t numNodes = 0;
        input >> numNodes;
        if( !input || (section->second.size() != numNodes+1) )
          DUNE_THROW( IOError, "Section 'Nodes' must contain exactly numNodes+1 lines." );

        std::vector< Node > nodes( numNodes );
        for( std::size_t i = 0; i < numNodes; ++i )
        {
          std::istringstream input( section->second[ i+1 ] );
          input >> nodes[ i ].id >> nodes[ i ].position[ 0 ] >> nodes[ i ].position[ 1 ] >> nodes[ i ].position[ 2 ];
          if( !input )
            DUNE_THROW( IOError, "Unable to read line " << (i+1) << " of 'Nodes' section" );
        }

        // sort nodes and ensure there are no duplicates
        std::sort( nodes.begin(), nodes.end(), [] ( const Node &a, const Node &b ) { return (a.id < b.id); } );
        const auto pos = std::adjacent_find( nodes.begin(), nodes.end(), [] ( const Node &a, const Node &b ) { return (a.id == b.id); } );
        if( pos != nodes.end() )
          DUNE_THROW( IOError, "Duplicate node " << pos->id << " in 'Nodes' section" );

        return std::move( nodes );
      }



      // parseMeshFormat
      // ---------------

      std::tuple< double, Format, std::size_t > parseMeshFormat ( const SectionMap &sectionMap )
      {
        SectionMap::const_iterator section = findUniqueSection( sectionMap, "MeshFormat" );
        if( section->second.size() != std::size_t( 1 ) )
          DUNE_THROW( IOError, "Section 'MeshFormat' must consist of exactly one line" );

        double version = 0.0;
        std::size_t fileType = 0, floatSize = 0;
        std::istringstream input( section->second.front() );
        input >> version >> fileType >> floatSize;
        if( !input )
          DUNE_THROW( IOError, "Unable to parse section 'MeshFormat'" );

        switch( fileType )
        {
        case 0:
          return std::make_tuple( version, ascii, floatSize );
        case 1:
          return std::make_tuple( version, binary, floatSize );
        default:
          DUNE_THROW( IOError, "Invalid file type: " << fileType );
        }
      }



      // readFile
      // --------

      SectionMap readFile ( const std::string &filename )
      {
        std::ifstream input( filename );

        SectionMap sectionMap;
        SectionMap::iterator section = sectionMap.end();
        while( input )
        {
          std::string line;
          getline( input, line );

          if( line.empty() )
            continue;

          if( line.front() == '$' )
          {
            if( line.substr( 1, 3 ) != "End" )
            {
              // start a new section
              if( section != sectionMap.end() )
                DUNE_THROW( IOError, "Unterminated Gmsh section: '" << section->first << "'" );
              section = sectionMap.emplace( line.substr( 1, line.npos ), Section() );
            }
            else
            {
              if( section == sectionMap.end() )
                DUNE_THROW( IOError, "End of unopened section '" << line.substr( 1, line.npos ) << "' encountered" );
              if( section->first != line.substr( 4, line.npos ) )
                DUNE_THROW( IOError, "Section '" << section->first << "' ended by '" << line.substr( 1, line.npos ) << "'" );
              section = sectionMap.end();
            }
          }
          else
          {
            if( section == sectionMap.end() )
              DUNE_THROW( IOError, "Data outside of section encountered" );
            section->second.emplace_back( std::move( line ) );
          }
        }
        if( section != sectionMap.end() )
          DUNE_THROW( IOError, "Unterminated Gmsh section: '" << section->first << "'" );

        return std::move( sectionMap );
      }



      // tags
      // ----

      std::vector< int > tags ( const std::vector< Element > &elements, const std::vector< std::size_t > &ids, std::size_t tag )
      {
        std::vector< int > tags;
        tags.reserve( ids.size() );
        for( std::size_t id : ids )
        {
          const auto pos = std::lower_bound( elements.begin(), elements.end(), id, [] ( const Element &a, std::size_t b ) { return (a.id < b); } );
          if( (pos == elements.end()) || (pos->id != id) )
            DUNE_THROW( RangeError, "No such element: " << id );
          if( tag >= pos->numTags )
            DUNE_THROW( RangeError, "Element " << id << " does not have tag " << tag );
          tags.push_back( pos->tags[ tag ] );
        }
        return std::move( tags );
      }



      // vertices
      // --------

      std::vector< std::size_t > vertices ( const std::vector< DuneEntity > &entities )
      {
        // collect all used vertices
        std::vector< std::size_t > vertices;
        for( const DuneEntity &entity : entities )
          vertices.insert( vertices.end(), entity.vertices().begin(), entity.vertices().end() );

        // remove duplicates
        std::sort( vertices.begin(), vertices.end() );
        vertices.erase( std::unique( vertices.begin(), vertices.end() ), vertices.end() );
        return std::move( vertices );
      }

    } // namespace Gmsh

  } // namespace Vem

} // namespace Dune
