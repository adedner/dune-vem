#ifndef DUNE_VEM_IO_GMSH_HH
#define DUNE_VEM_IO_GMSH_HH

#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace Dune
{

  namespace Vem
  {

    namespace Gmsh
    {

      typedef std::vector< std::string > Section;
      typedef std::multimap< std::string, Section > SectionMap;

      enum Format { ascii = 0, binary = 1 };

      std::tuple< double, Format, std::size_t > parseMeshFormat ( const SectionMap &sectionMap );

      SectionMap readFile ( const std::string &filename );

    } // namespace Gmsh

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_IO_GMSH_HH
