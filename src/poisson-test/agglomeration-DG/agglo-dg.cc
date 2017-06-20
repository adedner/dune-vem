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
#include <dune/fem/gridpart/leafgridpart.hh>
//#include <dune/fem/gridpart/adaptiveleafgridpart.hh>

#include <dune/fem/io/file/vtkio.hh>
#include <dune/fem/misc/mpimanager.hh>
#include <dune/fem/operator/linear/spoperator.hh>
#include <dune/fem/solver/cginverseoperator.hh>


#include <dune/fem/misc/h1norm.hh>
#include <dune/fem/misc/l2norm.hh>

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
#include "vemscheme.hh"

#include "problems/anisotropic.hh"
#include "problems/cosinusproduct.hh"
#include "problems/sinusproduct.hh"
#include "problems/cosinusproductmixedbc.hh"
#include "problems/reentrantcorner.hh"
#include "problems/sphereproblem.hh"
#include "problems/curvedridges.hh"


namespace Gmsh = Dune::Vem::Gmsh;

//typedef Dune::UGGrid< 2 > Grid;
typedef Dune::ALUGrid< 2, 2, Dune::cube, Dune::nonconforming > Grid;
// typedef Dune::ALUGrid< 2, 2, Dune::simplex, Dune::nonconforming > Grid;


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
  Dune::Fem::Parameter::append( "./data/agglomeration-parameter" );

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


  std::cout << " agglomeration.size() " << agglomeration.size() << std::endl;
  // define a function space type
  typedef Dune::Fem::FunctionSpace< GridPart::ctype, double, GridPart::dimension, 1 > FunctionSpaceType;



  // * * * Poisson problem solution * * * //

  typedef DiffusionModel< FunctionSpaceType, GridPart > ModelType;
  typedef typename ModelType::ProblemType ProblemType ;
  std::unique_ptr< ProblemType > problemPtr;
  const std::string problemNames [] = { "cos", "sphere", "sin", "mixedcos", "corner", "curvedridges", "anisotropic" };
  const int problemNumber = Dune::Fem::Parameter::getEnum( "poisson.problem", problemNames );

  switch( problemNumber )
  {
  case 0:
    problemPtr.reset( new CosinusProduct< FunctionSpaceType >() );
    break;

  case 1:
    problemPtr.reset( new SphereProblem< FunctionSpaceType >() );
    break;

  case 2:
    problemPtr.reset( new SinusProduct< FunctionSpaceType >() );
    break;

  case 3:
    problemPtr.reset( new CosinusProductMixedBC< FunctionSpaceType >() );
    break;

  case 4:
    problemPtr.reset( new ReentrantCorner< FunctionSpaceType >() );
    break;

  case 5:
    problemPtr.reset( new CurvedRidges< FunctionSpaceType >() );
    break;

  case 6:
    problemPtr.reset( new AnisotropicProblem< FunctionSpaceType >() );
    break;

  default:
    std::cerr << "Error: No problem selected. Set parameter 'poisson.problem'." << std::endl;
    return 1;
  }
  assert( problemPtr );
  ProblemType& problem = *problemPtr ;

  // implicit model for left hand side
  ModelType implicitModel( problem, gridPart );

  std::cout << "going in femscheme" << std::endl;
#if 0
  typedef Dune::Vem::AgglomerationDGSpace < FunctionSpaceType, GridPart, POLORDER > DiscreteFunctionSpaceType;
  typedef FemScheme< DiscreteFunctionSpaceType, ModelType > SchemeType;
#else
  typedef Dune::Vem::AgglomerationVEMSpace< FunctionSpaceType, GridPart, POLORDER > DiscreteFunctionSpaceType;
  typedef VemScheme< DiscreteFunctionSpaceType, ModelType > SchemeType;
#endif
  DiscreteFunctionSpaceType space( agglomeration );
  SchemeType scheme( space, implicitModel );

  typedef Dune::Fem::GridFunctionAdapter< ProblemType, GridPart > GridExactSolutionType;
  GridExactSolutionType gridExactSolution("exact solution", problem, gridPart, 4 );

  // setup the right hand side
  scheme.prepare();
  // solve once (assemble matrix)
  scheme.solve(true);

  // VTK output

  Dune::Fem::VTKIO< GridPart > vtkIO( gridPart, Dune::VTK::nonconforming );
  vtkIO.addVertexData( scheme.solution() );
  vtkIO.write( "./output/agglo-dg-poisson", Dune::VTK::ascii );


  // calculate standard error
  // select norm for error computation
  Dune::Fem::L2Norm< GridPart > normL2( gridPart );
  Dune::Fem::H1Norm< GridPart > normH1( gridPart );

  const double errorL2 = normL2.distance( gridExactSolution, scheme.solution() );
  const double errorH1 = normH1.distance( gridExactSolution, scheme.solution() );

  std::cout << "L2Norm error = " << errorL2 << std::endl;
  std::cout << "H1Norm error = " << errorH1 << std::endl;

  std::ofstream errorOut( "error.dat", std::ios_base::app );
  errorOut << agglomeration.size() << "    " << errorL2 << "    " << errorH1 << std::endl;

  std::cout << "Finished!" << std::endl;
  return 0;
}
catch( const Dune::Exception &e )
{
  std::cout << e << std::endl;
  return 1;
}
