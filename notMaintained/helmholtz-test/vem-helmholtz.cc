#define POLORDER 1
#define GRIDDIM 2
#define WORLDDIM 2
#define WANT_ISTL 0

#include <config.h>
#include <cmath>
#include <cstddef>
#include <complex>

// iostream includes
#include <iostream>
#include <memory>
#include <vector>

#include <dune/common/exceptions.hh>
#include <dune/alugrid/grid.hh>
//#include <dune/fem/function/adaptivefunction.hh>

// include grid part
//#include <dune/fem/gridpart/adaptiveleafgridpart.hh>
#include <dune/fem/gridpart/leafgridpart.hh>

#include <dune/fem/io/file/vtkio.hh>
#include <dune/fem/misc/mpimanager.hh>
//#include <dune/fem/operator/linear/spoperator.hh>
//#include <dune/fem/solver/cginverseoperator.hh>
// include output

#include <dune/fem/misc/h1norm.hh>
#include <dune/fem/misc/l2norm.hh>
#include <dune/fem/io/file/dataoutput.hh>

// vem includes
#include <dune/vem/agglomeration/agglomeration.hh>
#include <dune/vem/agglomeration/dgspace.hh>
#include <dune/vem/function/simple.hh>
#include <dune/vem/operator/mass.hh>
#include <dune/vem/io/gmsh.cc>


// include header of elliptic solver
#include "vemscheme.hh"
//#include "probleminterface.hh"
#include "poisson.hh"

namespace Gmsh = Dune::Vem::Gmsh;


typedef Dune::ALUGrid< 2, 2, Dune::cube, Dune::nonconforming > Grid;
//typedef Dune::ALUGrid< 2, 2, Dune::simplex, Dune::nonconforming > Grid;

// assemble-solve-estimate-mark-refine-IO-error-doitagain

template <class Grid, class AgglomerationType>
double algorithm ( Grid &grid, AgglomerationType &agglomeration )
{

  int step = 1;

  // we want to solve the problem on the leaf elements of the grid
  typedef Dune::Fem::LeafGridPart< Grid> GridPartType;
  GridPartType gridPart(grid);

  // use a scalar function space
  typedef Dune::Fem::FunctionSpace< double, std::complex<double>,
      GridPartType::dimension, 1 > FunctionSpaceType;

  // type of the mathematical model used
  typedef DiffusionModel< FunctionSpaceType, GridPartType > ModelType;

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

  // poisson solver
  typedef VemScheme< ModelType > VemSchemeType;
  VemSchemeType vemscheme( gridPart, implicitModel, agglomeration );


  typedef Dune::Fem::GridFunctionAdapter< ProblemType, GridPartType > GridExactSolutionType;
  GridExactSolutionType gridExactSolution("exact solution", problem, gridPart, 5 );
  //! input/output tuple and setup datawritter
  typedef std::tuple< const typename VemSchemeType::DiscreteFunctionType *, GridExactSolutionType * > IOTupleType;
  typedef Dune::Fem::DataOutput< Grid, IOTupleType > DataOutputType;
  IOTupleType ioTuple( &(vemscheme.solution()), &gridExactSolution) ; // tuple with pointers
  DataOutputType dataOutput( grid, ioTuple, DataOutputParameters( step ) );

  // setup the right hand side
  vemscheme.prepare();
  // solve once
  vemscheme.solve( true );

  // write initial solve
  dataOutput.write();

  // calculate error
  double errorl2 = 0 ;
  double errorh1 = 0 ;

  {
    // calculate standard error
    // select norm for error computation
    Dune::Fem::L2Norm< GridPartType > normL2( gridPart );
    Dune::Fem::H1Norm< GridPartType > normH1( gridPart );

    const double errorL2 = normL2.distance( gridExactSolution, vemscheme.solution() );
    const double errorH1 = normH1.distance( gridExactSolution, vemscheme.solution() );

    std::cout << errorL2 << " " << errorH1 << std::endl;
  }

  return 0 ;
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
  Dune::Fem::Parameter::append( "/home/gcd3/Project/dune-vem/src/helmholtz-test/data/fem-parameter" );

  // create grid from DGF file
  const std::string gridkey = Dune::Fem::IOInterface::defaultGridKey( Grid::dimension );
  const std::string gridfile = Dune::Fem::Parameter::getValue< std::string >( gridkey );

  // the method rank and size from MPIManager are static
  if( Dune::Fem::MPIManager::rank() == 0 )
    std::cout << "Loading macro grid: " << gridfile << std::endl;

  // copied from our agglomeration code..
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
  typedef Dune::Fem::LeafGridPart< Grid > GridPart;
  GridPart gridPart( *grid );

  // calculate first step
  Dune::Vem::Agglomeration< GridPart > agglomeration( gridPart, agglomerateIndices );
  double oldError = algorithm( *grid, agglomeration );
  return 0;
}
catch( const Dune::Exception &exception )
{
  std::cerr << "Error: " << exception << std::endl;
  return 1;
}
