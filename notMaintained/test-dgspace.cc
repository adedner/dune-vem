#define ALUGRID_CUBE
#define WORLDDIM 2
#define GRIDDIM 2

#include <config.h>

#include <cmath>
#include <cstddef>

#include <iostream>
#include <memory>
#include <vector>

#include <dune/common/exceptions.hh>

#if HAVE_UG
#include <dune/grid/uggrid.hh>
#endif

#include <dune/fem/function/adaptivefunction.hh>
#include <dune/fem/gridpart/leafgridpart.hh>
#include <dune/fem/io/file/vtkio.hh>
#include <dune/fem/misc/mpimanager.hh>
#include <dune/fem/operator/linear/spoperator.hh>
#include <dune/fem/solver/cginverseoperator.hh>

#include <dune/vem/agglomeration/agglomeration.hh>
#include <dune/vem/agglomeration/dgspace.hh>
#include <dune/vem/function/simple.hh>
#include <dune/vem/operator/mass.hh>
#include <dune/vem/io/gmsh.hh>

namespace Gmsh
{
  using namespace Dune::Vem::Gmsh;
}

int main ( int argc, char **argv )
try
{
  Dune::Fem::MPIManager::initialize( argc, argv );

  if( argc <= 1 )
  {
    std::cerr << "Usage: " << argv[ 0 ] << " <msh file>" << std::endl;
    return 1;
  }

#if HAVE_DUNE_ALUGRID
#warning USE ALUGrid
  typedef Dune::GridSelector::GridType Grid;
#elif HAVE_UG
#warning UG
  typedef Dune::UGGrid< GRIDDIM > Grid;
#else
#warning USE YaspGrid because nothing better was found
  typedef Dune::YaspGrid< GRIDDIM > Grid;
#endif

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

  // create DG space on agglomeration

  typedef Dune::Fem::FunctionSpace< GridPart::ctype, double, GridPart::dimension, 1 > FunctionSpace;
  typedef Dune::Vem::AgglomerationDGSpace< FunctionSpace, GridPart, 1 > DiscreteFunctionSpace;
  DiscreteFunctionSpace dgSpace( agglomeration );

  // initialize solution

  typedef Dune::Fem::AdaptiveDiscreteFunction< DiscreteFunctionSpace > DiscreteFunction;
  DiscreteFunction solution( "solution", dgSpace );

  // assemble right hand side

  const auto exactSolution
    = Dune::Vem::simpleFunction< FunctionSpace::DomainType >( [] ( const FunctionSpace::DomainType &x ) {
      double value = 1.0;
      for( int k = 0; k < GridPart::dimension; ++k )
        value *= std::sin( M_PI * x[ k ] );
      return Dune::FieldVector< double, 1 >( value );
    } );

  typedef Dune::Fem::GridFunctionAdapter< decltype( exactSolution ), GridPart > GridExactSolution;
  GridExactSolution gridExactSolution( "exact solution", exactSolution, gridPart, dgSpace.order()+1 );

  DiscreteFunction rhs( "right hand side", dgSpace );
  Dune::Vem::applyMass( gridExactSolution, rhs );

  // assemble mass matrix

  typedef Dune::Fem::SparseRowLinearOperator< DiscreteFunction, DiscreteFunction > LinearOperator;
  LinearOperator assembledMassOp( "assembled mass operator", dgSpace, dgSpace );

  Dune::Vem::MassOperator< LinearOperator > massOp( dgSpace );
  massOp.jacobian( solution, assembledMassOp );

  // solve

  Dune::Fem::CGInverseOperator< DiscreteFunction > invOp( assembledMassOp, 1e-8, 1e-8 );
  invOp( rhs, solution );

  // VTK output

  Dune::Fem::VTKIO< GridPart > vtkIO( gridPart, Dune::VTK::nonconforming );
  vtkIO.addVertexData( solution );
  vtkIO.write( "test-dgspace", Dune::VTK::ascii );

  return 0;
}
catch( const Dune::Exception &e )
{
  std::cout << e << std::endl;
  return 1;
}
