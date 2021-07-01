#include <config.h>

#include <cmath>
#include <cstddef>

#include <iostream>
#include <memory>
#include <vector>

#include <dune/common/exceptions.hh>

//#include <dune/grid/uggrid.hh>

#include <dune/alugrid/grid.hh>

#include <dune/fem/function/adaptivefunction.hh>
//#include <dune/fem/gridpart/leafgridpart.hh>
#include <dune/fem/gridpart/adaptiveleafgridpart.hh>

#include <dune/fem/io/file/vtkio.hh>
#include <dune/fem/misc/mpimanager.hh>
#include <dune/fem/operator/linear/spoperator.hh>
#include <dune/fem/solver/cginverseoperator.hh>

#include <dune/vem/agglomeration/agglomeration.hh>
#include <dune/vem/agglomeration/dgspace.hh>
#include <dune/vem/function/simple.hh>
#include <dune/vem/operator/mass.hh>
#include <dune/vem/io/gmsh.cc>

// includes for solving poisson problem //
#include <dune/fem/io/file/dataoutput.hh>

#define POLORDER 1
#define GRIDDIM 2
#define WORLDDIM 2
#define WANT_ISTL 0

#include "femscheme.hh"
#include "poisson.hh"


namespace Gmsh
{
using namespace Dune::Vem::Gmsh;
}

//typedef Dune::UGGrid< 2 > Grid;
typedef Dune::ALUGrid< 2, 2, Dune::cube, Dune::nonconforming > Grid;


int main ( int argc, char **argv )
try
{
  Dune::Fem::MPIManager::initialize( argc, argv );

  // append overloaded parameters from the command line
  Dune::Fem::Parameter::append( argc, argv );

  // append possible given parameter files
  for( int i = 1; i < argc; ++i )
    Dune::Fem::Parameter::append( argv[ i ] );

  // append default parameter file
  Dune::Fem::Parameter::append( "data/parameter" );

  const std::string gridkey = Dune::Fem::IOInterface::defaultGridKey( Grid::dimension );
  const std::string gridfile = Dune::Fem::Parameter::getValue< std::string >( gridkey );

  // the method rank and size from MPIManager are static
  if( Dune::Fem::MPIManager::rank() == 0 )
    std::cout << "Loading macro grid: " << gridfile << std::endl;

  //  const auto sectionMap = Gmsh::readFile( argv[ 1 ] );
  const auto sectionMap = Gmsh::readFile (gridfile);
  const auto nodes = Gmsh::parseNodes( sectionMap );
  const auto elements = Gmsh::parseElements( sectionMap );

  const auto entities = Gmsh::duneEntities( elements, Grid::dimension );
  const std::vector< std::size_t > vertices = Gmsh::vertices( entities );




  Dune::GridFactory< Grid > factory;
  Gmsh::insertVertices( factory, vertices, nodes );
  Gmsh::insertElements( factory, entities, vertices );

  std::unique_ptr< Grid > grid( factory.createGrid() );

  std::vector< std::size_t > elementIds = Gmsh::elements( grid->leafGridView(), factory, entities ); // contains the element ids of the original grid
  std::vector< int > agglomerateIndices = Gmsh::tags( elements, elementIds, 3 ); // contains the ids of individual element in each agglomerated polygon
  std::transform( agglomerateIndices.begin(), agglomerateIndices.end(), agglomerateIndices.begin(),
      [] ( int i ) { return i-1; } ); // This will give you correct number of agglomerations!
  // create grid part and agglomeration

  typedef Dune::Fem::AdaptiveLeafGridPart< Grid > GridPart;
  GridPart gridPart( *grid );

  Dune::Vem::Agglomeration< GridPart > agglomeration( gridPart, agglomerateIndices );


  std::cout << " agglomeration.size() " << agglomeration.size() << std::endl;
  // define a function space type
  typedef Dune::Fem::FunctionSpace< GridPart::ctype, double, GridPart::dimension, 1 > FunctionSpaceType;



  // * * * Poisson problem solution * * * //

  typedef DiffusionModel< FunctionSpaceType, GridPart > ModelType;
  typedef typename ModelType::ProblemType ProblemType ;
  ProblemType* problemPtr = 0 ;
  const std::string problemNames [] = { "cos", "sphere", "sin", "corner", "curvedridges" };
  const int problemNumber = Dune::Fem::Parameter::getEnum("poisson.problem", problemNames, 0 );

  switch ( problemNumber )
  {
  case 0:
    problemPtr = new CosinusProduct< FunctionSpaceType > ();
    break ;
  case 1:
    problemPtr = new SphereProblem< FunctionSpaceType > ();
    break ;
  case 2:
    problemPtr = new SinusProduct< FunctionSpaceType > ();
    break ;
  case 3:
    problemPtr = new ReentrantCorner< FunctionSpaceType > ();
    break ;
  case 4:
    problemPtr = new CurvedRidges< FunctionSpaceType > ();
    break ;
  default:
    problemPtr = new CosinusProduct< FunctionSpaceType > ();
  }
  assert( problemPtr );
  ProblemType& problem = *problemPtr ;

  // implicit model for left hand side
  ModelType implicitModel( problem, gridPart );

  std::cout << "going in femscheme" << std::endl;
  typedef FemScheme< ModelType > SchemeType;
  SchemeType scheme( gridPart, implicitModel, agglomeration );

  typedef Dune::Fem::GridFunctionAdapter< ProblemType, GridPart > GridExactSolutionType;
  GridExactSolutionType gridExactSolution("exact solution", problem, gridPart, 5 );

  //! input/output tuple and setup datawritter
  typedef std::tuple<const typename SchemeType::DiscreteFunctionType *, GridExactSolutionType * > IOTupleType;
  typedef Dune::Fem::DataOutput< Grid, IOTupleType > DataOutputType;
  IOTupleType ioTuple( &(scheme.solution()), &gridExactSolution) ; // tuple with pointers



  // setup the right hand side
  scheme.prepare();
  // solve once (assemble matrix)
  scheme.solve(true);

  // VTK output

  Dune::Fem::VTKIO< GridPart > vtkIO( gridPart, Dune::VTK::nonconforming );
  vtkIO.addVertexData( scheme.solution() );
  vtkIO.write( "test-poisson", Dune::VTK::ascii );


  // calculate standard error
  // select norm for error computation
  typedef Dune::Fem::L2Norm< GridPart > NormType;
  NormType norm( gridPart );
  std::cout << norm.distance( gridExactSolution, scheme.solution() ) << std::endl;
  std::cout << "Finished!" << std::endl;
  return 0;



}
catch( const Dune::Exception &e )
{
  std::cout << e << std::endl;
  return 1;
}
