#include <config.h>

#include <cmath>
#include <cstddef>

#include <iostream>
#include <memory>
#include <vector>

#include <dune/common/exceptions.hh>

#include <dune/grid/uggrid.hh>

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
#include <dune/vem/io/gmsh.cc>


namespace Gmsh
{
using namespace Dune::Vem::Gmsh;
}

typedef Dune::UGGrid< 2 > Grid;


//! [Implement a local function to use with a LocalFunctionAdapter]
template< class F1,class F2 >
struct LocalError
{
	// extract type of discrete function space
	typedef typename F1::DiscreteFunctionSpaceType DiscreteFunctionSpaceType;

	// extract type of function space
	typedef typename DiscreteFunctionSpaceType::FunctionSpaceType FunctionSpaceType;

	// extract type of grid part
	typedef typename DiscreteFunctionSpaceType::GridPartType GridPartType;

	// extract type of element (entity of codimension 0)
	typedef typename GridPartType::template Codim< 0 >::EntityType EntityType;

	// type of range values
	typedef typename FunctionSpaceType::RangeType RangeType;

	// constructor
	LocalError ( const F1 &f1, const F2 &f2 )
	: lf1_( f1 ),
	  lf2_( f2 )
	{}

	// evaluate local function
	template< class PointType >
	void evaluate ( const PointType &x, RangeType &val )
	{
		RangeType tmp;
		lf1_.evaluate( x, val );
		lf2_.evaluate( x, tmp );
		val -= tmp;
	}

	// initialize to new entity
	void init ( const EntityType &entity )
	{
		lf1_.init( entity );
		lf2_.init( entity );
	}

private:
	typename F1::LocalFunctionType lf1_;
	typename F2::LocalFunctionType lf2_;
};
//! [Implement a local function to use with a LocalFunctionAdapter]


template< class GridFunctionU, class GridFunctionV >
double computeError( const GridFunctionU &u, const GridFunctionV &v )
{
	// extract grid part
	// (and silently assume this also holds for GridFunctionV)
	typedef typename GridFunctionU::GridPartType GridPartType;
	const GridPartType &gridPart = u.gridPart();

	// initialize error (squared) to zero
	double error = 0;

	// obtian types of local functions
	typedef typename GridFunctionU::LocalFunctionType LocalFunctionUType;
	typedef typename GridFunctionV::LocalFunctionType LocalFunctionVType;

	// obtain type of iterator, entity and geometry
	typedef typename GridPartType::template Codim< 0 >::IteratorType IteratorType;
	typedef typename GridPartType::template Codim< 0 >::EntityType EntityType;
	typedef typename GridPartType::template Codim< 0 >::GeometryType GeometryType;

	// define type of quadrature to be used
	typedef Dune::Fem::CachingQuadrature< GridPartType, 0 > QuadratureType;

	// loop over the grid
	const IteratorType end = gridPart.template end< 0 >();
	for( IteratorType it = gridPart.template begin< 0 >(); it != end; ++it )
	{
		// obtain a reference to the entity
		const EntityType &entity = *it;

		// obtain geometry
		const GeometryType geometry = entity.geometry();
		//		std::cout << " " << std::endl;
		//		std::cout << geometry.center() << std::endl;

		// get the local representation of both functions on this entity...
		const LocalFunctionUType uLocal = u.localFunction( entity );
		const LocalFunctionVType vLocal = v.localFunction( entity );

		// get the correct quadrature order
		const int order = uLocal.order() + vLocal.order() + vLocal.order(); // see the comment ** on pg 82 (VEm-1 book).
		//		std::cout << " quadr order = " << order << std::endl;
		// setup quadrature and loop over points
		QuadratureType quadrature( entity, order );
		for( unsigned int np = 0; np < quadrature.nop(); ++np )
		{
			// evaluate both grid functions
			typename LocalFunctionUType::RangeType uValue;
			uLocal.evaluate( quadrature[ np ], uValue );

			typename LocalFunctionVType::RangeType vValue;
			vLocal.evaluate( quadrature[ np ], vValue );

			// compute the norm square of the difference between both values
			const double diff = (uValue - vValue).two_norm2();
			//			std::cout << diff << std::endl;
			// multiply by quadrature weight and integration element |DF_T| and sum up
			const double integrationElement = geometry.integrationElement( quadrature.point( np ) );
			error += diff * quadrature.weight( np ) * integrationElement;
		}
	}

	// finally take the square root
	return std::sqrt( error );
}



int main ( int argc, char **argv )
try
{
	Dune::Fem::MPIManager::initialize( argc, argv );

	static const bool assembled = 1;

	//  if( argc <= 1 )
	//  {
	//    std::cerr << "Usage: " << argv[ 0 ] << " <msh file>" << std::endl;
	//    return 1;
	//  }

	// read gmsh file

	//  const auto sectionMap = Gmsh::readFile( argv[ 1 ] );
	//	const auto sectionMap = Gmsh::readFile ("/home/gcd3/codes/grids/unitsq_unstr.msh");
	const auto sectionMap = Gmsh::readFile ("./gmsh/mymesh.msh");
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

	// create grid part and agglomeration

	typedef Dune::Fem::LeafGridPart< Grid > GridPart;
	GridPart gridPart( *grid );

	Dune::Vem::Agglomeration< GridPart > agglomeration( gridPart, agglomerateIndices );

	// create DG space on agglomeration

	typedef Dune::Fem::FunctionSpace< GridPart::ctype, double, GridPart::dimension, 1 > FunctionSpace;
	typedef Dune::Vem::AgglomerationDGSpace< FunctionSpace, GridPart, 1 > DiscreteFunctionSpace;
	DiscreteFunctionSpace dgSpace( gridPart, agglomeration );

	// initialize solution
	std::ofstream fRhs ("./rhs-agglo.dat");
	std::ofstream fSoln ("./soln-agglo.dat");

	typedef Dune::Fem::AdaptiveDiscreteFunction< DiscreteFunctionSpace > DiscreteFunction;
	DiscreteFunction solution( "solution", dgSpace );

	// assemble right hand side

	const auto exactSolution
	= Dune::Vem::simpleFunction< FunctionSpace::DomainType >( [] ( const FunctionSpace::DomainType &x ) {
		double value = 1.0;
		for( int k = 0; k < GridPart::dimension; ++k ){
			value *= std::sin( M_PI * x[ k ] );

		}
		//		std::cout << Dune::FieldVector< double, 1 >( value ) << std::endl;
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
	rhs.print(fRhs);
	Dune::Fem::CGInverseOperator< DiscreteFunction > invOp( assembledMassOp, 1e-8, 1e-8 );
	invOp( rhs, solution );
	solution.print(fSoln);

	// VTK output

	Dune::Fem::VTKIO< GridPart > vtkIO( gridPart, Dune::VTK::nonconforming );
	vtkIO.addVertexData( solution );
	vtkIO.write( "test-dgspace", Dune::VTK::ascii );


	//! [Use a local function Adapter to output the error on the grid]
	// use a local function adapter to compute u - uh (result is a grid function)
	typedef LocalError< DiscreteFunction, GridExactSolution > LocalErrorType;
	typedef Dune::Fem::LocalFunctionAdapter< LocalErrorType > ErrorFunctionType;
	LocalErrorType localError( solution, gridExactSolution );
	ErrorFunctionType errorFunction( "error", localError, gridPart );

	// setup tuple of functions to output (discrete solution, exact solution, and difference)
	typedef Dune::tuple< const DiscreteFunction *, GridExactSolution *, ErrorFunctionType * > IOTupleType;
	IOTupleType ioTuple( &solution, &gridExactSolution, &errorFunction );

	double error = 0;
	error = computeError( solution, gridExactSolution );
	std::cout << "Error in l2 projection: " << error << std::endl;
	return 0;
}
catch( const Dune::Exception &e )
{
	std::cout << e << std::endl;
	return 1;
}
