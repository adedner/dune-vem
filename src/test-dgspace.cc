#include <config.h>

#include <dune/common/exceptions.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/grid/uggrid.hh>

#include <dune/vem/io/gmsh.hh>

namespace Gmsh
{
  using namespace Dune::Vem::Gmsh;
}

typedef Dune::UGGrid< 2 > Grid;

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
  std::vector< int > agglomerateIndices = Gmsh::tags( elements, elementIds, 4 );

  typedef Dune::Fem::LeafGridPart< Grid > GridPart;

  GridPart gridPart( *grid );

  Dune::Vem::Agglomeration< GridPart > agglomeration( gridPart, agglomerateIndices );


  typedef Dune::Fem::FunctionSpace< GridPart::ctype, double, GridPart::dimension, 1 > FunctionSpace;
  typedef Dune::Vem::AgglomerateDGSpace< GridPart, FunctionSpace, 2 > DiscreteFunctionSpace;
  DiscreteFunctionSpace dgSpace( gridPart, agglomeration );

  return 0;
}
catch( const Dune::Exception &e )
{
  std::cout << e << std::endl;
  return 1;
}
