#ifndef DUNE_VEM_IO_GMSH_HH
#define DUNE_VEM_IO_GMSH_HH

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <utility>

#include <dune/common/fvector.hh>

#include <dune/geometry/type.hh>

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
        std::size_t id = 0;
        GeometryType type;
        std::unique_ptr< std::size_t[] > vertices;
      };

      std::vector< DuneEntity > duneEntities ( const std::vector< Element > &elements, unsigned int dim );

      template< class Iterator >
      Iterator findNode ( Iterator begin, Iterator end, const std::size_t nodeId );

      std::vector< Element > parseElements ( const SectionMap &sectionMap );
      std::vector< Node > parseNodes ( const SectionMap &sectionMap );
      std::tuple< double, Format, std::size_t > parseMeshFormat ( const SectionMap &sectionMap );

      SectionMap readFile ( const std::string &filename );

      std::vector< Node > vertices ( const std::vector< Element > &elements, const std::vector< Node > &nodes, unsigned int dim );

    } // namespace Gmsh

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_IO_GMSH_HH
