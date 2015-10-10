#include <cassert>

#include <fstream>
#include <sstream>

#include <dune/common/exceptions.hh>

#include <dune/vem/io/gmsh.hh>

namespace Dune
{

  namespace Vem
  {

    namespace Gmsh
    {

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



      // parseNodes
      // ----------

      std::vector< Node > parseNodes ( const SectionMap &sectionMap )
      {
        if( sectionMap.count( "Nodes" ) != std::size_t( 1 ) )
          DUNE_THROW( IOError, "A Gmsh file requires exactly one 'Nodes' section" );

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
          input >> nodes[ i ].first >> nodes[ i ].second[ 0 ] >> nodes[ i ].second[ 1 ] >> nodes[ i ].second[ 2 ];
          if( !input )
            DUNE_THROW( IOError, "Unable to read line " << (i+1) << " of 'Nodes' section" );
        }

        // sort nodes and ensure there are no duplicates
        std::sort( nodes.begin(), nodes.end(), [] ( const Node &a, const Node &b ) { return (a.first < b.first); } );
        const auto pos = std::adjacent_find( nodes.begin(), nodes.end(), [] ( const Node &a, const Node &b ) { return (a.first == b.first); } );
        if( pos != nodes.end() )
          DUNE_THROW( IOError, "Duplicate node " << pos->first << " in 'Nodes' section" );

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

    } // namespace Gmsh

  } // namespace Vem

} // namespace Dune
