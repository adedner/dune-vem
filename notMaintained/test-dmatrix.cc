#include <config.h>

#include <cmath>
#include <cstddef>

#include <iostream>
#include <memory>
#include <vector>

#include <dune/common/exceptions.hh>

#include <dune/grid/common/rangegenerators.hh>
//#include <dune/grid/uggrid.hh>

#include <dune/alugrid/grid.hh>

#include <dune/fem/gridpart/leafgridpart.hh>
#include <dune/fem/misc/mpimanager.hh>

#include <dune/vem/agglomeration/agglomeration.hh>
#include <dune/vem/agglomeration/indexset.hh>
#include <dune/vem/agglomeration/dgspace.hh>
#include <dune/vem/io/gmsh.cc>

#include <dune/grid/io/file/gmshwriter.hh>


#include <dune/fem/function/adaptivefunction.hh>
#include <dune/fem/gridpart/leafgridpart.hh>
#include <dune/fem/io/file/vtkio.hh>
#include <dune/fem/misc/mpimanager.hh>
#include <dune/fem/operator/linear/spoperator.hh>
#include <dune/fem/solver/cginverseoperator.hh>


#include <dune/common/dynvector.hh>
#include <dune/common/dynmatrix.hh>
#include <dune/geometry/referenceelements.hh>


#define POLORDER 1
#define GRIDDIM 2
#define WORLDDIM 2
#define WANT_ISTL 0


template< class Enumerator >
void printSet ( std::ostream &out, std::size_t size, Enumerator enumerator )
{
  char delim = '{';
  for( std::size_t i = 0; i < size; ++i, delim = ',' )
    out << delim << " " << enumerator( i );
  out << " }";
}

namespace Gmsh
{
using namespace Dune::Vem::Gmsh;
}

//typedef Dune::UGGrid< 2 > Grid;
typedef Dune::ALUGrid< 2, 2, Dune::cube, Dune::nonconforming > Grid;


template <class GridPart>
double algorithm ( GridPart &gridPart, std::vector<int>agglomerateIndices)
{
  double error = 0;

  Dune::Vem::Agglomeration< GridPart > agglomeration( gridPart, agglomerateIndices );


  // create DG space on agglomeration //
  // define a function space type
  typedef Dune::Fem::FunctionSpace< typename GridPart::ctype, double, GridPart::dimension, 1 > FunctionSpace;
  typedef Dune::Vem::AgglomerationDGSpace< FunctionSpace, GridPart, 1 > DiscreteFunctionSpace;
  DiscreteFunctionSpace dfSpace( gridPart, agglomeration );


  // write some typedefs first:
  typedef typename DiscreteFunctionSpace::RangeType RangeType;
  typedef typename DiscreteFunctionSpace::JacobianRangeType JacobianRangeType;
  typedef typename DiscreteFunctionSpace::IteratorType IteratorType;
  typedef typename IteratorType::Entity EntityType;
  //  typedef typename EntityType::EntityPointer EntityPointerType;
  typedef typename EntityType::Geometry GeometryType;

  const unsigned int numDofs = dfSpace.blockMapper().maxNumDofs()*DiscreteFunctionSpace::localBlockSize ;
  std::vector< RangeType > phi( numDofs );
  std::vector< JacobianRangeType > dphi( numDofs );
  static const int dimDomain = GridPart::dimension;


  typedef typename GridPart::IndexSetType LeafIndexSet;
  const LeafIndexSet& lset = gridPart.indexSet();

  const IteratorType end = dfSpace.end();

  // from dune/fem/gridpart/test/checkseed.hh
  // could be useful.. do not remove
  typedef typename GridPart::template Codim< 0 >::EntityPointerType EntityPointerType;
  typedef typename GridPart::template Codim< 0 >::EntitySeedType EntitySeedType;

  // get the iterator for elements
  //  https://github.com/bempp/dune-alugrid/blob/master/dune/alugrid/common/writeparalleldgf.hh
  typedef typename GridPart::template Codim< 0 >::IteratorType ElementIterator;
  typedef typename ElementIterator::Entity Element ;
  typedef typename Element::EntityPointer ElementPointer;
  typedef typename Element::EntitySeed ElementSeed ;
  std::vector< ElementSeed > elementSeeds;
  std::vector <std::vector<ElementSeed>> PolygonalMeshIDs;

  PolygonalMeshIDs.resize(agglomeration.size());

  int myElem = 0;
  int currentPolygon = -1;

  typedef typename GridPart::IndexSetType IndexSetType;
  typedef typename GridPart::IndexSetType::IndexType IndexType;
  const IndexSetType & indexSet = gridPart.indexSet();
  IndexType maxIndex = 0;

  typedef typename GridPart::IntersectionIteratorType IntersectionIteratorType;
  typedef typename IntersectionIteratorType::Intersection IntersectionType;
  typedef typename IntersectionType::Geometry IntersectionGeometryType;

  typedef Dune::DynamicMatrix<int> Matrix;
  typedef Dune::DynamicVector<int> ScalarField;
  std::vector<int>NVertexVector;
  NVertexVector.resize(agglomeration.size() );
  NVertexVector = {0};


  std::ofstream fDlocal ("/home/gcd3/codes/dune-vem/output/Dmatrix.dat");

  // create agglomeration index set
  Dune::Vem::AgglomerationIndexSet< GridPart > agIndexSet( agglomeration );

  int count = 0 ;


  for( const auto &element : Dune::elements( static_cast< typename GridPart::GridViewType >( gridPart ), Dune::Partitions::interiorBorder ) )
  {
    const auto geometry = element.geometry();
    const EntityType &entity = *element;
    currentPolygon = agglomeration.index(*element); // the polygon we are integrating
    elementSeeds.push_back( entity.seed() ) ;
    PolygonalMeshIDs[currentPolygon].push_back(entity.seed() );
    std::vector<int> Vector1; // Vector containing local index of the vertices of the polygon that are in element T. Remember this vector will alway be of size
    std::vector<int> Vector2; // Vector containing Global index of the vertex of the polygon
    // geometry.corners();
    Vector1.resize(geometry.corners());
    Vector1 = {0}; // initialise

    Dune::GeometryType gt = geometry.type();
    auto& refElement = Dune::ReferenceElements<double,dimDomain>::general(gt);
    const int LeafElementIndex = lset.index(entity); // Global element number of the current element

    NVertexVector[currentPolygon] =  agIndexSet.numPolyVertices(element,GridPart::dimension) ;
  }


  // matrix calculation step
  Matrix Dlocal;
  Matrix PI1;
  Matrix DlocalTranspose;
  Matrix DTDinversed;
  // some more typedefs... these are copied from the femscheme.hh file
  typedef Dune::Fem::AdaptiveDiscreteFunction< DiscreteFunctionSpace > DiscreteFunctionType;
  typedef typename DiscreteFunctionType::LocalFunctionType LocalFunctionType;
  typedef typename DiscreteFunctionSpace::DomainType DomainType;
  typedef typename DiscreteFunctionSpace::BasisFunctionSetType BasisFunctionSetType;


  // loop over number of polygons
  currentPolygon = 0;

  double AreaofDomain = 0.0;


  for(auto PolygonIterator = PolygonalMeshIDs.begin();PolygonIterator!=PolygonalMeshIDs.end();++PolygonIterator) {

    int numPolygonVertices = NVertexVector [currentPolygon] ;
    double AreaofPolygon = 0.0;
    Dlocal.resize(numPolygonVertices, numDofs);
    DlocalTranspose.resize(numDofs, numPolygonVertices) ;
    PI1.resize(numDofs, numPolygonVertices) ;
    DTDinversed.resize (numPolygonVertices, numPolygonVertices);
    std::pair< Dune::FieldVector< typename GridPart::ctype, GridPart::dimensionworld >, Dune::FieldVector< typename GridPart::ctype, GridPart::dimensionworld > > PolygonBBox ;


    // iterate over sub-elements within each polygon
    for(auto it = (*PolygonIterator).begin();it!=(*PolygonIterator).end();++it) {

      typedef typename ElementIterator::Entity::Geometry LeafGeometry;

      const ElementPointer ep = gridPart.grid().entityPointer( *it );

      const LeafGeometry geo = ep->geometry();

      Dune::GeometryType gt = ep->type();

      auto& ref = Dune::ReferenceElements<double,dimDomain>::general(gt);

      auto &basis = dfSpace.basisFunctionSet( ep ) ;

      AreaofPolygon = AreaofPolygon + geo.volume ();

      int numElemVertices = geo.corners();

      std::vector< RangeType > Phi ( numDofs );

      for (int i=0; i< numElemVertices; ++i){

        if (agIndexSet.localIndex( ep, i, GridPart::dimension )!= -1) {


          Dune::FieldVector<double,GridPart::dimension> vX = geo.corner(i);

          int LocalDofInPolygon = ref.subEntity(0, 0 , i, GridPart::dimension) ;

          Dune::FieldVector<double, GridPart::dimension> global(vX);

          const Dune::FieldVector< double, GridPart::dimension > &VertexlocalCoordinate = ref.position( i, GridPart::dimension );

          basis.evaluateAll( VertexlocalCoordinate, Phi );

          // assemble D-matrix
          for (int jCol = 0; jCol < numDofs; ++jCol) {
            Dlocal[LocalDofInPolygon][jCol] =  Phi[jCol] ;
          }

        }
      }


    } // finished iterating over sub-elements in a given polygon


    // write out the D-matrix
    fDlocal << currentPolygon << std::endl;
    for (int iRow = 0; iRow < numPolygonVertices; ++iRow) {
      for (int jCol = 0; jCol < numDofs; ++jCol) {
        fDlocal << iRow << ", " << jCol << " , " << Dlocal[iRow][jCol] << std::endl;
      }
    }
    fDlocal << " " << std::endl;

    AreaofDomain = AreaofDomain + AreaofPolygon ;
    ++currentPolygon;

  }


  return error; // yet to code
}



int main ( int argc, char **argv )
try
{
  Dune::Fem::MPIManager::initialize( argc, argv );

  //  if( argc <= 1 )
  //  {
  //    std::cerr << "Usage: " << argv[ 0 ] << " <msh file>" << std::endl;
  //    return 1;
  //  }

  // read gmsh file

  const auto sectionMap = Gmsh::readFile( "/home/gcd3/codes/dune-vem/data/partitioned-mesh.msh" );
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

  algorithm(gridPart,agglomerateIndices);

  // write the grid you originally imported
  typedef Grid::LeafGridView GV;
  Dune::GmshWriter<typename Grid::LeafGridView> writer(grid->leafGridView());
  writer.write("/home/gcd3/codes/dune-vem/output/yourmesh.msh");

  std::cout << "Done..!" << std::endl;
  return 0;
}
catch( const Dune::Exception &e )
{
  std::cout << e << std::endl;
  return 1;
}
