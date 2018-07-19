#define POLORDER 1
#define GRIDDIM 2
#define WORLDDIM 2
#define WANT_ISTL 0

#include <config.h>
#include <cmath>
#include <cstddef>
// iostream includes
#include <iostream>

#include <memory>
#include <vector>

#include <dune/common/exceptions.hh>
#include <dune/alugrid/grid.hh>

#include <dune/fem/function/adaptivefunction.hh>

// include grid part
//#include <dune/fem/gridpart/adaptiveleafgridpart.hh>
#include <dune/fem/gridpart/leafgridpart.hh>

#include <dune/fem/io/file/vtkio.hh>
#include <dune/fem/misc/mpimanager.hh>
#include <dune/fem/operator/linear/spoperator.hh>
#include <dune/fem/solver/cginverseoperator.hh>


#include <dune/fem/misc/h1norm.hh>
#include <dune/fem/misc/l2norm.hh>

// include output
#include <dune/fem/io/file/dataoutput.hh>

// vem includes
#include <dune/vem/agglomeration/agglomeration.hh>
#include <dune/vem/agglomeration/dgspace.hh>
#include <dune/vem/function/simple.hh>
#include <dune/vem/operator/mass.hh>
#include <dune/vem/io/gmsh.cc>

// include header for heat model
#include "heat.hh"

#include "heatmodel.hh"
#include "heatscheme.hh"



//#include "femscheme.hh"
#include "vemscheme.hh"
//
namespace Gmsh = Dune::Vem::Gmsh;
//
typedef Dune::ALUGrid< 2, 2, Dune::simplex, Dune::nonconforming > Grid;


// assemble-solve-estimate-mark-refine-IO-error-doitagain
template <class Grid, class AgglomerationType>
double algorithm ( Grid &grid, AgglomerationType &agglomeration )
{
  int step = 1;

  // create time provider
  Dune::Fem::GridTimeProvider< Grid > timeProvider( grid );

  // we want to solve the problem on the leaf elements of the grid
  //typedef Dune::Fem::AdaptiveLeafGridPart< Grid, Dune::InteriorBorder_Partition > GridPartType;
  typedef Dune::Fem::LeafGridPart< Grid > GridPartType;
  GridPartType gridPart(grid);

  typedef Dune::Fem::FunctionSpace< double, double, GridPartType::dimension, 1 > FunctionSpaceType;


  // type of the mathematical model used
  typedef TimeDependentCosinusProduct< FunctionSpaceType > ProblemType;
  typedef HeatModel< FunctionSpaceType, GridPartType > ModelType;

  ProblemType problem( timeProvider ) ;

  // implicit model for left hand side
  ModelType implicitModel( problem, gridPart, true );

  // explicit model for right hand side
  ModelType explicitModel( problem, gridPart, false );

  // create heat scheme
  typedef HeatScheme< ModelType, ModelType > SchemeType;
  SchemeType scheme( gridPart, implicitModel, explicitModel, agglomeration );

  typedef Dune::Fem::GridFunctionAdapter< ProblemType, GridPartType > GridExactSolutionType;
  GridExactSolutionType gridExactSolution("exact solution", problem, gridPart, 5 );
  //! input/output tuple and setup datawritter
  typedef std::tuple< const typename SchemeType::DiscreteFunctionType *, GridExactSolutionType * > IOTupleType;
  typedef Dune::Fem::DataOutput< Grid, IOTupleType > DataOutputType;
  IOTupleType ioTuple( &(scheme.solution()), &gridExactSolution) ; // tuple with pointers
  DataOutputType dataOutput( grid, ioTuple, DataOutputParameters( step ) );

  const double endTime  = Dune::Fem::Parameter::getValue< double >( "heat.endtime", 2.0 );
  const double dtreducefactor = Dune::Fem::Parameter::getValue< double >("heat.reducetimestepfactor", 1 );
  double timeStep = Dune::Fem::Parameter::getValue< double >( "heat.timestep", 0.125 );

  timeStep *= pow(dtreducefactor,step);

  //! [time loop]
  // initialize with fixed time step
  timeProvider.init( timeStep ) ;

  // initialize scheme and output initial data
  scheme.initialize();

  // time loop, increment with fixed time step
  for( ; timeProvider.time() < endTime; timeProvider.next( timeStep ) )
  //! [time loop]
  {
    dataOutput.write( timeProvider );
    // assemble explicit pare
    scheme.prepare();
    // solve once (we need to assemble the system the first time the method
    // is called)
    scheme.solve( (timeProvider.timeStep()==0) );
  }

  // output final solution
  dataOutput.write( timeProvider );

  // select norm for error computation
  typedef Dune::Fem::L2Norm< GridPartType > NormType;
  NormType norm( gridPart );
  return norm.distance( gridExactSolution, scheme.solution() );
}

// main
// ----

int main ( int argc, char **argv )
try
{
  // initialize MPI, if necessary
  Dune::Fem::MPIManager::initialize( argc, argv );

  // append overloaded parameters from the command line
  Dune::Fem::Parameter::append( argc, argv );

  // append possible given parameter files
  for( int i = 1; i < argc; ++i )
    Dune::Fem::Parameter::append( argv[ i ] );

  // append default parameter file
  Dune::Fem::Parameter::append( "./data/parameter" );

  // create grid from DGF file
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

  //typedef Dune::Fem::AdaptiveLeafGridPart< Grid > GridPart;
  typedef Dune::Fem::LeafGridPart< Grid > GridPart;
  GridPart gridPart( *grid );

  Dune::Vem::Agglomeration< GridPart > agglomeration( gridPart, agglomerateIndices );

  //// just solve once for now
  double oldError = algorithm( *grid, agglomeration);

  return 0;
}
catch( const Dune::Exception &exception )
{
  std::cerr << "Error: " << exception << std::endl;
  return 1;
}
