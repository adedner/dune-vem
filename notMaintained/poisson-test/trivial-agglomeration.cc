#include <config.h>

#include <cmath>
#include <cstddef>

#include <iostream>
#include <memory>
#include <vector>

#include <dune/common/exceptions.hh>

//#include <dune/grid/uggrid.hh>

#include <dune/alugrid/grid.hh>
#include <dune/alugrid/dgf.hh>

#include <dune/fem/function/adaptivefunction.hh>
#include <dune/fem/gridpart/leafgridpart.hh>
#include <dune/fem/io/file/dataoutput.hh>
#include <dune/fem/io/file/vtkio.hh>
#include <dune/fem/io/parameter.hh>
#include <dune/fem/misc/l2norm.hh>
#include <dune/fem/misc/mpimanager.hh>
#include <dune/fem/operator/linear/spoperator.hh>
#include <dune/fem/solver/cginverseoperator.hh>
#include <dune/fem/space/discontinuousgalerkin.hh>

#include <dune/vem/agglomeration/agglomeration.hh>
#include <dune/vem/agglomeration/dgspace.hh>

#include "poisson.hh"
#include "probleminterface.hh"

#include "model.hh"

#include "dgelliptic.hh"
#include "dgrhs.hh"

static constexpr int polynomialOrder = 1;

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

  const std::string gridkey = Dune::Fem::IOInterface::defaultGridKey( Grid::dimension );
  const std::string gridfile = Dune::Fem::Parameter::getValue< std::string >( gridkey );

  // the method rank and size from MPIManager are static
  if( Dune::Fem::MPIManager::rank() == 0 )
    std::cout << "Loading macro grid: " << gridfile << std::endl;
  Dune::GridPtr< Grid > grid( gridfile );

  // initial refinement
  const int level = Dune::Fem::Parameter::getValue< int >( "poisson.level", 0 );
  grid->globalRefine( level );

  // define a function space type
  typedef Dune::Fem::FunctionSpace< Grid::ctype, double, Grid::dimension, 1 > FunctionSpaceType;

  // setup poisson problem
  typedef ProblemInterface< FunctionSpaceType > Problem;
  std::unique_ptr< Problem > problem;
  const std::string problemNames [] = { "cos", "sphere", "sin", "corner", "curvedridges" };
  const int problemNumber = Dune::Fem::Parameter::getEnum( "poisson.problem", problemNames, 0 );
  switch ( problemNumber )
  {
  case 0:
    problem.reset( new CosinusProduct< FunctionSpaceType >() );
    break;

  case 1:
    problem.reset( new SphereProblem< FunctionSpaceType >() );
    break;

  case 2:
    problem.reset( new SinusProduct< FunctionSpaceType >() );
    break;

  case 3:
    problem.reset( new ReentrantCorner< FunctionSpaceType >() );
    break;

  case 4:
    problem.reset( new CurvedRidges< FunctionSpaceType >() );
    break;

  default:
    DUNE_THROW( Dune::Exception, "Unknown problem number: " << problemNumber );
  }
  assert( problem );

  double prevError = std::numeric_limits< double >::max();
  const int repeats = Dune::Fem::Parameter::getValue< int >( "poisson.repeats", 0 );
  for( int repeat = 0; repeat <= repeats; ++repeat )
  {
    // refine grid
    if( repeat > 0 )
      grid->globalRefine( Dune::DGFGridInfo< Grid >::refineStepsForHalf() );

    // create grid part
    typedef Dune::Fem::LeafGridPart< Grid > GridPart;
    GridPart gridPart( *grid );

    // convert exact solution to grid function
    typedef Dune::Fem::GridFunctionAdapter< Problem, GridPart > GridExactSolutionType;
    GridExactSolutionType gridExactSolution( "exact solution", *problem, gridPart, 5 );

    // setup trivial agglomeration
    const std::size_t size = gridPart.indexSet().size( 0 );
    std::vector< std::size_t > agglomerateIndices( size );
    for( std::size_t i = 0; i < size; ++i )
      agglomerateIndices[ i ] = i;

    Dune::Vem::Agglomeration< GridPart > agglomeration( gridPart, agglomerateIndices );

    // create discrete function space
    typedef Dune::Vem::AgglomerationDGSpace< FunctionSpaceType, GridPart, polynomialOrder > DiscreteFunctionSpace;
    DiscreteFunctionSpace dfSpace( agglomeration );

    // create solution
    typedef Dune::Fem::AdaptiveDiscreteFunction< DiscreteFunctionSpace > DiscreteFunction;
    DiscreteFunction solution( "solution", dfSpace );
    solution.clear();

    // create model
    typedef DiffusionModel< FunctionSpaceType, GridPart > Model;
    Model model( *problem, gridPart );

    // assemble right hand side
    DiscreteFunction rhs( "rhs", dfSpace );
    assembleDGRHS ( model, rhs );

    // assemble elliptic operator
    typedef Dune::Fem::SparseRowLinearOperator< DiscreteFunction, DiscreteFunction > LinearOperator;
    LinearOperator linearOp( "assembled elliptic operator", dfSpace, dfSpace );

    typedef DifferentiableDGEllipticOperator< LinearOperator, Model > EllipticOperator;
    EllipticOperator ellipticOp( model, dfSpace );
    ellipticOp.jacobian( solution, linearOp );

    // solve linear system
    typedef Dune::Fem::CGInverseOperator< DiscreteFunction > LinearInverseOperator;
    const double solverEps = Dune::Fem::Parameter::getValue< double >( "poisson.solvereps", 1e-8 );
    LinearInverseOperator invOp( linearOp, solverEps, solverEps );
    invOp( rhs, solution );

    // VTK output
    Dune::Fem::VTKIO< GridPart > vtkIO( gridPart, Dune::VTK::nonconforming );
    vtkIO.addVertexData( solution );
    vtkIO.write( "trivial-agglomeration-" + std::to_string( repeat ), Dune::VTK::ascii );

    // calculate standard error
    Dune::Fem::L2Norm< GridPart > norm( gridPart );
    double error = norm.distance( gridExactSolution, solution );
    std::cout << repeat << "  " << error << "  " << (repeat > 0 ? std::to_string( std::log( prevError / error ) / std::log( 2.0 ) ) : std::string( "---" )) << std::endl;
    prevError = error;
  }
  return 0;
}
catch( const Dune::Exception &e )
{
  std::cout << e << std::endl;
  return 1;
}
