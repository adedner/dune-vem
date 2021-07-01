#include <config.h>

#include <cmath>
#include <cstddef>

#include <iostream>
#include <memory>
#include <vector>

#include <dune/common/exceptions.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/grid/io/file/vtk.hh>
//#include <dune/grid/uggrid.hh>

#include <dune/alugrid/grid.hh>

#include <dune/vem/io/gmsh.hh>

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
  Dune::MPIHelper::instance( argc, argv );

  if( argc <= 1 )
  {
    std::cerr << "Usage: " << argv[ 0 ] << " <msh file>" << std::endl;
    return 1;
  }

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
  std::array< std::vector< int >, 4 > tags;
  for( int i = 0; i < 4; ++i )
    tags[ i ] = Gmsh::tags( elements, elementIds, i );

  Dune::VTKWriter< Grid::LeafGridView > vtkWriter( grid->leafGridView() );
  vtkWriter.addCellData( elementIds, "id" );
  for( int i = 0; i < 4; ++i )
    vtkWriter.addCellData( tags[ i ], "tag-" + std::to_string( i ) );
  vtkWriter.write( "test-gmsh" );

  return 0;
}
catch( const Dune::Exception &e )
{
  std::cout << e << std::endl;
  return 1;
}
