#ifndef DUNE_VEM_IO_GMSH_HH
#define DUNE_VEM_IO_GMSH_HH

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <utility>

#include <dune/common/fvector.hh>

namespace Dune
{

  namespace Vem
  {

    namespace Gmsh
    {

      enum Format { ascii = 0, binary = 1 };

      struct ElementType
      {
        std::size_t numNodes;
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

      std::vector< Element > parseElements ( const SectionMap &sectionMap );
      std::vector< Node > parseNodes ( const SectionMap &sectionMap );
      std::tuple< double, Format, std::size_t > parseMeshFormat ( const SectionMap &sectionMap );

      SectionMap readFile ( const std::string &filename );

    } // namespace Gmsh

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_IO_GMSH_HH
