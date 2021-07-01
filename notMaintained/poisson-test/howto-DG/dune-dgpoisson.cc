//=========================================================================
// Use this block, if you want operate with the GridSelector.
//#define UGGRID
#define ALUGRID_CUBE
#define GRIDDIM 2
#include <config.h>
#include <dune/grid/utility/gridtype.hh> // only for debuging
//=========================================================================

//#include <dune/grid/uggrid.hh>
#include <dune/alugrid/grid.hh>

// iostream includes
#include <iostream>

// include grid part
#include <dune/fem/function/adaptivefunction.hh>

#include <dune/fem/gridpart/adaptiveleafgridpart.hh>
//#include <dune/fem/gridpart/leafgridpart.hh>

// include output
#include <dune/fem/io/file/dataoutput.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/gmshwriter.hh>
#define POLORDER 1
#define WORLDDIM 2
#define WANT_ISTL 0

// include header of elliptic solver
#include "femscheme.hh"
#include "poisson.hh"


//typedef Dune::UGGrid< 2 > GridType;
typedef Dune::ALUGrid< 2, 2, Dune::cube, Dune::nonconforming > GridType;
// main
// ----
using namespace Dune;
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

  // type of hierarchical grid
  //  typedef Dune::GridSelector::GridType  HGridType ;

  // create grid from DGF file
  const std::string gridkey = Dune::Fem::IOInterface::defaultGridKey( GridType::dimension );
  const std::string gridfile = Dune::Fem::Parameter::getValue< std::string >( gridkey );

  // the method rank and size from MPIManager are static
  if( Dune::Fem::MPIManager::rank() == 0 )
    std::cout << "Loading macro grid: " << gridfile << std::endl;

  // alugrid

  typedef GridType::LeafGridView GV;
  std::auto_ptr<GridType> grid( Dune::GmshReader<GridType>::read( gridfile, true, true ) );
  const GV gv = grid->leafGridView();
  Dune::GmshWriter<typename GridType::LeafGridView> writer(grid->leafGridView());
  writer.write("./output/yourmesh.msh");

  // create grid part
  typedef Dune::Fem::AdaptiveLeafGridPart< GridType> GridPart;

  GridPart gridPart( *grid );


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

  // // // // // // // // // // // // // // // // // // // // // // // //

  // implicit model for left hand side
  ModelType implicitModel( problem, gridPart );

  std::cout << "going in femscheme" << std::endl;
  typedef FemScheme< ModelType > SchemeType;
  SchemeType scheme( gridPart, implicitModel);

  typedef Dune::Fem::GridFunctionAdapter< ProblemType, GridPart > GridExactSolutionType;
  GridExactSolutionType gridExactSolution("exact solution", problem, gridPart, 5 );

  //! input/output tuple and setup datawritter
  typedef std::tuple<const typename SchemeType::DiscreteFunctionType *, GridExactSolutionType * > IOTupleType;
  typedef Dune::Fem::DataOutput< GridType, IOTupleType > DataOutputType;
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
  typedef Dune::Fem::L2Norm< GridPart > L2NormType;

  L2NormType normL2( gridPart );

  std::cout << "L2Norm error = " << normL2.distance( gridExactSolution, scheme.solution() ) << std::endl;

    std::cout << "Finished!" << std::endl;
    return 0;

  // // // // // // // // // // // // // // // // // // // // // // // //


}
catch( const Dune::Exception &exception )
{
  std::cerr << "Error: " << exception << std::endl;
  return 1;
}
