#include <cassert>

#include <fstream>
#include <sstream>
#include <utility>

#include <dune/common/exceptions.hh>

#include <dune/vem/io/gmsh.hh>

namespace Dune
{

  namespace Vem
  {

    namespace Gmsh
    {

      // parseMeshFormat
      // ---------------

      std::tuple< double, Format, std::size_t > parseMeshFormat ( const SectionMap &sectionMap )
      {
        if( sectionMap.count( "MeshFormat" ) != std::size_t( 1 ) )
          DUNE_THROW( IOError, "A Gmsh file requires exactly one 'MeshFormat' section" );

        SectionMap::const_iterator section = sectionMap.find( "MeshFormat" );
        assert( section != sectionMap.end() );
        if( section->second.size() != std::size_t( 1 ) )
          DUNE_THROW( IOError, "Section 'MeshFormat' must consist of exactly one line" );

        double version = 0.0;
        std::size_t fileType = 0, floatSize = 0;
        std::istringstream input( sectionMap.find( "MeshFormat" )->second.front() );
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
