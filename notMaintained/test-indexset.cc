#include <config.h>

#include <cmath>
#include <cstddef>

#include <iostream>
#include <memory>
#include <vector>

#include <dune/common/exceptions.hh>

#include <dune/grid/common/rangegenerators.hh>
//#include <dune/grid/uggrid.hh>

#include <dune/alugrid/grid.hh>

#include <dune/fem/gridpart/leafgridpart.hh>
#include <dune/fem/misc/mpimanager.hh>

#include <dune/vem/agglomeration/agglomeration.hh>
#include <dune/vem/agglomeration/indexset.hh>
#include <dune/vem/io/gmsh.cc>

#include <dune/grid/io/file/gmshwriter.hh>

template< class Enumerator >
void printSet ( std::ostream &out, std::size_t size, Enumerator enumerator )
{
  char delim = '{';
  for( std::size_t i = 0; i < size; ++i, delim = ',' )
    out << delim << " " << enumerator( i );
  out << " }";
}


namespace Gmsh
{
  using namespace Dune::Vem::Gmsh;
}

typedef Dune::ALUGrid< 2, 2, Dune::cube, Dune::nonconforming > Grid;
//typedef Dune::ALUGrid< 2, 2, Dune::simplex, Dune::nonconforming > Grid;
//typedef Dune::UGGrid< 2 > Grid;

int main ( int argc, char **argv )
try
{
  Dune::Fem::MPIManager::initialize( argc, argv );

  if( argc <= 1 )
  {
    std::cerr << "Usage: " << argv[ 0 ] << " <msh file>" << std::endl;
    return 1;
  }

  // read gmsh file

  const auto sectionMap = Gmsh::readFile( argv[ 1 ] );
  const auto nodes = Gmsh::parseNodes( sectionMap );
  const auto elements = Gmsh::parseElements( sectionMap );

  const auto entities = Gmsh::duneEntities( elements, Grid::dimension );
  const std::vector< std::size_t > vertices = Gmsh::vertices( entities );

  Dune::GridFactory< Grid > factory;
  Gmsh::insertVertices( factory, vertices, nodes );
  Gmsh::insertElements( factory, entities, vertices );

  std::unique_ptr< Grid > grid( factory.createGrid() );

  std::vector< std::size_t > elementIds = Gmsh::elements( grid->leafGridView(), factory, entities );
  std::vector< int > agglomerateIndices = Gmsh::tags( elements, elementIds, 3 );
  std::transform( agglomerateIndices.begin(), agglomerateIndices.end(), agglomerateIndices.begin(), [] ( int i ) { return i-1; } );

  // create grid part and agglomeration

  typedef Dune::Fem::LeafGridPart< Grid > GridPart;
  GridPart gridPart( *grid );

  Dune::Vem::Agglomeration< GridPart > agglomeration( gridPart, agglomerateIndices );

  // write the grid you originally imported
  typedef Grid::LeafGridView GV;
  Dune::GmshWriter<typename Grid::LeafGridView> writer(grid->leafGridView());
  writer.write("../output/yourmesh.msh");

  // create agglomeration index set

  Dune::Vem::AgglomerationIndexSet< GridPart > agIndexSet( agglomeration );

  for( const auto &element : Dune::elements( static_cast< GridPart::GridViewType >( gridPart ), Dune::Partitions::interiorBorder ) )
  {
    const auto geometry = element.geometry();
    std::cout << "element: ";
    printSet( std::cout, geometry.corners(), [ &geometry ] ( std::size_t i ) { return geometry.corner( i ); } );
    std::cout << std::endl;

    for( int codim = Grid::dimension; codim <= Grid::dimension; ++codim )
    {
      std::cout << "codim " << codim << ": ";
      {
        const auto enumerator = [ &agIndexSet, &element, codim ] ( std::size_t i ) { return agIndexSet.subIndex( element, i, codim ); };
        printSet( std::cout, agIndexSet.subAgglomerates( element, codim ), enumerator );
      }
      std::cout << std::endl;
      {
        const auto enumerator = [ &agIndexSet, &element, codim ] ( std::size_t i ) { return agIndexSet.localIndex( element, i, codim ); };
        printSet( std::cout, 4, enumerator );
      }
      std::cout << std::endl;
    }
  }

  return 0;
}
catch( const Dune::Exception &e )
{
  std::cout << e << std::endl;
  return 1;
}
