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
#include <dune/vem/agglomeration/dgspace.hh>
#include <dune/vem/agglomeration/indexset.hh>
#include <dune/vem/io/gmsh.cc>

#include <dune/grid/io/file/gmshwriter.hh>


#include <dune/fem/function/adaptivefunction.hh>
#include <dune/fem/gridpart/leafgridpart.hh>
#include <dune/fem/io/file/vtkio.hh>
#include <dune/fem/misc/mpimanager.hh>
#include <dune/fem/operator/linear/spoperator.hh>
#include <dune/fem/solver/cginverseoperator.hh>


#include <dune/common/dynmatrix.hh>
#include <dune/common/dynvector.hh>
#include <dune/geometry/referenceelements.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/ilu.hh>
#include <dune/istl/io.hh>
#include <dune/istl/matrix.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/istl/matrixmatrix.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/superlu.hh>

#include <dune/geometry/quadraturerules.hh>


#define POLORDER 2
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

//! Diffusion coefficient
template< class DomainType, class DiffusionTensorType >
void evalDiffCoeff ( DomainType &x, DiffusionTensorType &Dcoeff )
{
  Dcoeff = 0;
  Dcoeff[ 0 ][ 0 ] = 1.0 + pow( x[ 1 ], 2 );
  Dcoeff[ 1 ][ 1 ] = 1.0 + pow( x[ 0 ], 2 );
  Dcoeff[ 0 ][ 1 ] = -x[ 0 ] * x[ 1 ] * std::sin( M_PI * x[ 0 ] ) * std::sin( M_PI * x[ 1 ] );
  Dcoeff[ 1 ][ 0 ] = Dcoeff[ 0 ][ 1 ];
}

//! Reaction coefficient
template< class DomainType, class RangeType >
void evalReactionCoeff ( DomainType &x, RangeType &ReactionCoeff )
{

  ReactionCoeff = pow( x[ 0 ], 2 ) + pow( x[ 1 ], 3 ) + 1.0;
}

// forcing term
template< class DomainType, class RangeType, class JacobianRangeType >
void evalForcingFunction ( DomainType &x, RangeType &truesolution, \
                           JacobianRangeType &truesolutionGradient, RangeType &reactioncoefficient, RangeType &forcingfunction )
{
  forcingfunction = pow( M_PI, 2 ) * truesolution * ( 2.0 + x[ 0 ]*x[ 0 ]+x[ 1 ]*x[ 1 ] );
  forcingfunction += x[ 1 ] * truesolution * truesolutionGradient[ 0 ][ 1 ];
  forcingfunction += x[ 0 ] * truesolution * truesolutionGradient[ 0 ][ 0 ];
  forcingfunction += 4.0 * pow( M_PI, 2 ) * x[ 0 ] * x[ 1 ] *truesolution *std::cos( M_PI * x[ 0 ] ) * std::cos( M_PI * x[ 1 ] );
  forcingfunction += reactioncoefficient * truesolution;
}

template< class DomainType, class RangeType, class JacobianRangeType >
void evalTrueSolution ( DomainType &x, RangeType &truesolution, JacobianRangeType &truesolutionGradient )
{

  // problem 1
  truesolution = sin( M_PI * x[ 0 ] ) * sin( M_PI * x[ 1 ] );
  truesolutionGradient[ 0 ][ 0 ] = M_PI * cos( M_PI * x[ 0 ] ) * sin( M_PI * x[ 1 ] );
  truesolutionGradient[ 0 ][ 1 ] = M_PI * sin( M_PI * x[ 0 ] ) * cos( M_PI * x[ 1 ] );
}

template< class DomainType, class RangeType, class T >
void evalMonomialBasis ( DomainType &x, DomainType &xE, RangeType hE, std::vector< T > &PhiMonomialBasis )
{
  PhiMonomialBasis = 0;

  for( int i = 0; i < PhiMonomialBasis.size(); ++i )
  {
    if( i == 0 )
      PhiMonomialBasis.push_back( 1.0 );
    else
    {
      double monomialValue = ( x[ i - 1 ] - xE[ i - 1 ] ) / hE;
      PhiMonomialBasis.push_back( monomialValue );
    }
  }
}


namespace Gmsh
{
  using namespace Dune::Vem::Gmsh;
}

//typedef Dune::UGGrid< 2 > Grid;
//typedef Dune::ALUGrid< 2, 2, Dune::simplex, Dune::nonconforming > Grid;
typedef Dune::ALUGrid< 2, 2, Dune::cube, Dune::nonconforming > Grid;


template< class GridPart >
double algorithm ( GridPart &gridPart, std::vector< int > agglomerateIndices )
{
  double error = 0;

  Dune::Vem::Agglomeration< GridPart > agglomeration( gridPart, agglomerateIndices );


  // create DG space on agglomeration //
  // define a function space type
  typedef Dune::Fem::FunctionSpace< typename GridPart::ctype, double, GridPart::dimension, 1 > FunctionSpace;
  typedef Dune::Vem::AgglomerationDGSpace< FunctionSpace, GridPart, POLORDER > DiscreteFunctionSpace;
  DiscreteFunctionSpace dfSpace( gridPart, agglomeration );


  // write some typedefs first:
  typedef typename DiscreteFunctionSpace::RangeType RangeType;
  typedef typename DiscreteFunctionSpace::JacobianRangeType JacobianRangeType;
  typedef typename DiscreteFunctionSpace::IteratorType IteratorType;
  typedef typename IteratorType::Entity EntityType;
  //  typedef typename EntityType::EntityPointer EntityPointerType;
  typedef typename EntityType::Geometry GeometryType;

  static const int dimDomain = GridPart::dimension;


  typedef typename GridPart::IndexSetType LeafIndexSet;
  const LeafIndexSet &lset = gridPart.indexSet();

  const IteratorType end = dfSpace.end();

  std::cout << "check " << dfSpace.size() << std::endl;

  // from dune/fem/gridpart/test/checkseed.hh
  // could be useful.. do not remove
  typedef typename GridPart::template Codim< 0 >::EntityPointerType EntityPointerType;
  typedef typename GridPart::template Codim< 0 >::EntitySeedType EntitySeedType;

  // get the iterator for elements
  //  https://github.com/bempp/dune-alugrid/blob/master/dune/alugrid/common/writeparalleldgf.hh
  typedef typename GridPart::template Codim< 0 >::IteratorType ElementIterator;
  typedef typename ElementIterator::Entity Element;
  typedef typename Element::EntityPointer ElementPointer;
  typedef typename Element::EntitySeed ElementSeed;
  std::vector< ElementSeed > elementSeeds;
  std::vector< std::vector< ElementSeed > > PolygonalMeshIDs;

  std::vector< int > PolygonalMeshVertexIDs;
  std::vector< int > PolygonalMeshEdgeIDs;

  PolygonalMeshIDs.resize( agglomeration.size());

  int myElem = 0;
  int currentPolygon = -1;
  double DiffusionTensorTrace = 0;
  std::vector< int > Vector1;     // Vector containing local index of the vertices of the polygon that are in element T. Remember this vector will alway be of size
  std::vector< int > Vector2;     // Vector containing Global index of the vertex of the polygon
  typedef Dune::FieldMatrix< RangeType, dimDomain, dimDomain > DiffusionTensorType;
  //  typedef Dune::FieldVector<RangeType, dimDomain> IntegrationPoint;

  typedef typename GridPart::IndexSetType IndexSetType;
  typedef typename GridPart::IndexSetType::IndexType IndexType;
  const IndexSetType &indexSet = gridPart.indexSet();
  IndexType maxIndex = 0;

  typedef typename GridPart::IntersectionIteratorType IntersectionIteratorType;
  typedef typename IntersectionIteratorType::Intersection IntersectionType;
  typedef typename IntersectionType::Geometry IntersectionGeometryType;


  typedef Dune::DynamicMatrix< double > Matrix;
  typedef Dune::DynamicVector< double > ScalarField;
  std::vector< int > NVertexVector;
  NVertexVector.resize( agglomeration.size() );
  NVertexVector = {0};
  const int numNodesinOriginalMesh = gridPart.indexSet().size( GridPart::dimension );
  int numVtxPolygonalMesh;
  double AverageMeshSize = 0;
  //  PolygonalMeshVertexIDs.resize(numVtxPolygonalMesh); //



  std::ofstream fDofInfo( "../../output/dofinfo.dat" );
  std::ofstream fGaussPts( "../../output/gaussPoints.dat" );
  std::ofstream fDlocal( "../../output/Dmatrix.dat" );
  std::ofstream fClocal( "../../output/Cmatrix.dat" );
  std::ofstream fDlocalT( "../../output/DTmatrix.dat" );
  std::ofstream fDTD( "../../output/DTD.dat" );
  std::ofstream fDTDI( "../../output/DTDI.dat" );
  std::ofstream fTemp( "../../output/temp.dat" );
  std::ofstream fTemp2( "../../output/temp2.dat" );
  std::ofstream fTemp3( "../../output/temp3.dat" );
  std::ofstream fPI1( "../../output/PI1.dat" );
  std::ofstream fPI_PHI_1( "../../output/PI_PHI_1.dat" );
  std::ofstream fPI0X( "../../output/PI0X.dat" );
  std::ofstream fKclocal( "../../output/Kc.dat" );
  std::ofstream fJunk1( "../../output/junk1.dat" );

  std::ofstream fAmatrix( "../../output/Amatrix.dat" );
  std::ofstream fXVemSoln( "../../output/xVemsoln.dat" );
  std::ofstream fbmatrix( "../../output/bmatrix.dat" );






  std::ofstream fpolyFaceInfo( "../../output/polygonal_faces.dat" );
  std::ofstream fpolyConn( "../../output/polygonal_conn.dat" );
  std::ofstream fpolyvert( "../../output/polygonal_vertices.dat" );
  std::ofstream fpolyCoM( "../../output/polygonal_CoM" );
  std::ofstream fOut;
  fOut.open( "../../output/out.dat", std::ofstream::out | std::ofstream::app );
  // create agglomeration index set
  Dune::Vem::AgglomerationIndexSet< GridPart > agIndexSet( agglomeration );
  const int nk = ( POLORDER + 1 ) * ( POLORDER + 2 ) / 2;
  const int numDofsPerEdge = POLORDER - 1;
  const int numTermsin_k_minus_2 = ( POLORDER - 1 ) * ( POLORDER ) / 2;      // see pg 260, RD Cook, Fig. 7.1-2
  int numLocalDof = 0;

  int count = 0;
  int oldPolygon = -1;


  for( const auto &element : Dune::elements( static_cast< typename GridPart::GridViewType >( gridPart ), Dune::Partitions::interiorBorder ) )
  {
    const auto geometry = element.geometry();
    const EntityType &entity = *element;

    currentPolygon = agglomeration.index( *element );           // the polygon we are integrating

    elementSeeds.push_back( entity.seed() );
    PolygonalMeshIDs[ currentPolygon ].push_back( entity.seed() );

    // geometry.corners();
    Vector1.resize( geometry.corners());

    Dune::GeometryType gt = geometry.type();
    auto &refElement = Dune::ReferenceElements< double, dimDomain >::general( gt );
    const int LeafElementIndex = lset.index( entity );           // Global element number of the current element

    NVertexVector[ currentPolygon ] =  agIndexSet.numPolyVertices( element, GridPart::dimension );
    Vector2.resize( agIndexSet.numPolyVertices( element, GridPart::dimension ));


    for( int codim = GridPart::dimension; codim <= GridPart::dimension; ++codim )
    {

      {

        for( int k = 0; k < geometry.corners(); ++k )
          Vector1[ k ] = agIndexSet.localIndex( element, k, codim );


        for( int j = 0; j < agIndexSet.subAgglomerates( element, codim ); ++j )
          Vector2[ j ] = agIndexSet.subIndex( element, j, codim );


      }

    }

    if( currentPolygon != oldPolygon )

      for( int iRow = 0; iRow < NVertexVector[ currentPolygon ]; ++iRow )
      {
        // just storing the global vertex numbers for the polygonal mesh in PolygonalMeshVertexIDs
        PolygonalMeshVertexIDs.push_back( agIndexSet.subIndex( element, iRow, GridPart::dimension ));                         // = agIndexSet.subIndex( element, iRow, GridPart::dimension ) ;
        PolygonalMeshEdgeIDs.push_back( agIndexSet.subIndex( element, iRow, GridPart::dimension - 1 ));                         // store all edge ids

        oldPolygon = currentPolygon;
        ++count;
      }


    // * * * *  W R I T E  P O L Y G O N A L  M E S H  D A T A  * * * *

    fpolyConn << currentPolygon;
    for( int idum = 0; idum < NVertexVector[ currentPolygon ]; ++idum )
      fpolyConn << " " << agIndexSet.subIndex( element, idum, GridPart::dimension ) << " ";
    fpolyConn << std::endl;

    for( int j = 0; j <  geometry.corners(); ++j )
    {
      int LocalVertexIndexofPolygon = Vector1[ j ];
      if( LocalVertexIndexofPolygon != -1 )
      {
        int GlobalVertexIndexofPolygon = Vector2[ LocalVertexIndexofPolygon ];
        fpolyvert <<  GlobalVertexIndexofPolygon << " " << geometry.corner( j )  << std::endl;
      }

    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  }


  // sort and get the unique vertex index for the polygonal mesh
  std::sort( PolygonalMeshVertexIDs.begin(), PolygonalMeshVertexIDs.end() );
  PolygonalMeshVertexIDs.erase( unique( PolygonalMeshVertexIDs.begin(), PolygonalMeshVertexIDs.end() ), PolygonalMeshVertexIDs.end() );
  numVtxPolygonalMesh = PolygonalMeshVertexIDs.size();

  // sort and get the unique vertex index for the edges in polygonal mesh
  std::sort( PolygonalMeshEdgeIDs.begin(), PolygonalMeshEdgeIDs.end() );
  PolygonalMeshEdgeIDs.erase( unique( PolygonalMeshEdgeIDs.begin(), PolygonalMeshEdgeIDs.end() ), PolygonalMeshEdgeIDs.end() );
  const int numEdgesPolygonalMesh = PolygonalMeshEdgeIDs.size();

  const int numTotalVertexDofs = numVtxPolygonalMesh;
  const int numTotalEdgeDofs = numEdgesPolygonalMesh * ( POLORDER - 1 );


  //  for (int iRow = 0; iRow <

  // matrix calculation step
  Matrix Dlocal;
  Matrix Clocal;
  Matrix PI1;
  Matrix PI1T;
  Matrix PI_PHI_1;
  Matrix DlocalTranspose;
  Matrix DTDinversed;
  Matrix Kmat;      // the Kln matrix
  Matrix Amatrix;
  Matrix PI0X;       // better name?
  Matrix PI0XTransposed;
  Matrix Imatrix;


  double H0;       // this is only for linear case, we still need to code H_l

  Matrix PI0Xt_x_Kmat;
  Matrix KConsistencyLocal;       // consistency part of the local stiffness matrix
  Matrix KStbilityLocal;       // stability part of the local stiffness matrix
  Matrix KStiffLocal;       // local stiffness matrix
  Matrix MassLocal;       // local mass matrix
  Matrix PI1T_x_ReactionMatrix;
  Matrix ReactionMatrix;       // reaction term integrated over polygon (will be nk x nk)
  Matrix I_minus_PI_PHI_1;
  Matrix I_minus_PI_PHI_1_Transposed;
  Matrix KStiffGlobal;       // Global stiffness matrix
  ScalarField f_load_global;
  ScalarField RhsVector;
  ScalarField F_mAlpha_local;
  ScalarField f_load_local;

  // global soln matrices
  ScalarField x_soln_global;
  Matrix A_lhs_global;
  ScalarField b_rhs_global;
  Matrix A_lhs_global_inverse;
  std::vector< Matrix > vec_mat_PI1;




  std::vector< int > LocalPolygonalConnectivity;

  std::vector< int > LocalEdgeIDs;

  std::vector< int > LocalVertexDofs;      // stores the vertex dofs for a given polygon
  std::vector< int > LocalEdgeDofs;      // stores the edge dofs for a given polygon
  std::vector< int > LocalInternalDofs;      // stores the internal dofs for a given polygon
  std::vector< int > LocalDofVector;      // stores the total dofs for a given polygon



  std::vector< std::vector< int > > PolygonalMeshConnectivityArray;
  std::vector< std::vector< int > > GlobalDofArray;




  // some more typedefs... these are copied from the femscheme.hh file
  typedef Dune::Fem::AdaptiveDiscreteFunction< DiscreteFunctionSpace > DiscreteFunctionType;
  typedef typename DiscreteFunctionType::LocalFunctionType LocalFunctionType;
  typedef typename DiscreteFunctionSpace::DomainType DomainType;
  typedef typename DiscreteFunctionSpace::BasisFunctionSetType BasisFunctionSetType;


  // loop over number of polygons
  currentPolygon = 0;
  oldPolygon = -1;

  double AreaofDomain = 0.0;

  KStiffGlobal.resize( numVtxPolygonalMesh, numVtxPolygonalMesh );
  f_load_global.resize( numVtxPolygonalMesh );
  f_load_global = 0;
  KStiffGlobal = 0;


  for( auto PolygonIterator = PolygonalMeshIDs.begin(); PolygonIterator != PolygonalMeshIDs.end(); ++PolygonIterator )
  {

    std::cout << "procsing polygon " << currentPolygon  << std::endl;
    int numPolygonVertices = NVertexVector [ currentPolygon ];

    int numVertexDofs = numPolygonVertices;
    int numEdgeDofs = numPolygonVertices * ( POLORDER - 1 );
    int numInternalDofs = numTermsin_k_minus_2;
    numLocalDof = numVertexDofs + numEdgeDofs + numInternalDofs;
    //LocalDofVector.resize(numLocalDof);
    //LocalEdgeDofs.resize(numEdgeDofs);
    //LocalInternalDofs.resize(numInternalDofs);

    double AreaofPolygon = 0.0;
    double CenterOfMass_x = 0.0;
    double CenterOfMass_y = 0.0;
    double MeanDiffusionCoefficient = 0.0;
    double MeanReactionCoefficient = 0.0;
    double StabilityScalingFactor = 0.0;
    Dlocal.resize( numLocalDof, nk );
    Clocal.resize( nk, numLocalDof );
    DlocalTranspose.resize( nk, numLocalDof );
    Kmat.resize( dimDomain, dimDomain );
    PI1.resize( nk, numLocalDof );
    PI1T.resize( numLocalDof, nk );
    f_load_local.resize( numLocalDof );
    PI_PHI_1.resize( numLocalDof, numLocalDof );
    DTDinversed.resize( nk, nk );
    PI0X.resize( GridPart::dimension, numLocalDof );
    //
    MassLocal.resize( numLocalDof, numLocalDof );
    ReactionMatrix.resize( nk, nk );

    // rhs
    F_mAlpha_local.resize( nk );           // A vector containing \int (f m\alpha)_E

    DiffusionTensorType DiffusionCoefficent;
    RangeType ReactionCoefficient;
    RangeType forcingfunction = 0.0;
    JacobianRangeType TrueSolutionGradient = 0;
    RangeType TrueSolution = 0;
    MeanDiffusionCoefficient = 0.0;
    F_mAlpha_local = 0.0;
    int numSubElements = 0;
    double dummyArea = 0;
    // iterate over sub-elements within each polygon
    for( auto it = ( *PolygonIterator ).begin(); it != ( *PolygonIterator ).end(); ++it )
    {

      typedef typename ElementIterator::Entity::Geometry LeafGeometry;

      const ElementPointer ep = gridPart.grid().entityPointer( *it );
      ++numSubElements;
      const LeafGeometry geo = ep->geometry();
      Dune::GeometryType gt = ep->type();
      auto &ref = Dune::ReferenceElements< double, dimDomain >::general( gt );
      const int LeafElementIndex = lset.index( ep );                 // Global element number of the current element
      auto &basis = dfSpace.basisFunctionSet( ep );
      const int numBasisFct = basis.size();
      AreaofPolygon = AreaofPolygon + geo.volume();
      int numElemVertices = geo.corners();


      CenterOfMass_x = CenterOfMass_x + geo.center()[ 0 ];
      CenterOfMass_y = CenterOfMass_y + geo.center()[ 1 ];

      // storing the local edge info
      // We do: for a given element T, go to each edge of T, find if it belongs to the polygon boundary, if yes,
      // find the local edge number in the polygon (not in T!) and then store it in an array
      LocalEdgeIDs.resize( numPolygonVertices );
      for( int iRow = 0; iRow < numElemVertices; ++iRow )
      {
        int LocalEdgeId = agIndexSet.localIndex( ep, iRow, GridPart::dimension-1 );
        if( LocalEdgeId != -1 )
        {

          int GlobalEdgeId = agIndexSet.subIndex( ep, LocalEdgeId,  GridPart::dimension-1 );
          std::cout << currentPolygon << " " << LocalEdgeId <<  " "<< GlobalEdgeId << std::endl;

        }

      }

      if( currentPolygon != oldPolygon )
      {
        // LocalPolygonalConnectivity renamed to LocalVertexDofs
        LocalPolygonalConnectivity.resize( numPolygonVertices );

        // The following will simply store the connectivity which can be accssed correctly only by element pointer and a local vertex number in reference element of Element in the underlying grid (T)
        for( int iRow = 0; iRow < numPolygonVertices; ++iRow )
        {
          LocalVertexDofs.push_back( agIndexSet.subIndex( ep, iRow, GridPart::dimension ) );
          LocalPolygonalConnectivity[ iRow ] = agIndexSet.subIndex( ep, iRow, GridPart::dimension );
          LocalEdgeIDs[ iRow ] = agIndexSet.subIndex( ep, iRow, GridPart::dimension-1 );
          for( int jCol = 0; jCol < numDofsPerEdge; ++jCol )
          {
            LocalEdgeDofs.push_back( numVtxPolygonalMesh + LocalEdgeIDs[ iRow ]  * numDofsPerEdge + jCol );
            fDofInfo << "poly " << currentPolygon << "edge = " << LocalEdgeIDs[ iRow ] << ", edgedof = " << numVtxPolygonalMesh + ( LocalEdgeIDs[ iRow ] - 1 ) * numDofsPerEdge + jCol + 1 << std::endl;
          }
        }

        for( int jCol = 0; jCol < numInternalDofs; ++jCol )
          LocalInternalDofs.push_back( numVtxPolygonalMesh + numTotalEdgeDofs + currentPolygon * numInternalDofs + jCol );


        // [ LocalVertexDofs | LocalEdgeDofs | LocalInternalDofs];
        GlobalDofArray.push_back( LocalVertexDofs );                       //
        GlobalDofArray.push_back( LocalEdgeDofs );
        GlobalDofArray.push_back( LocalInternalDofs );

        PolygonalMeshConnectivityArray.push_back( LocalPolygonalConnectivity );                       // stores vertex connectivity only
        oldPolygon = currentPolygon;
      }

      Dune::FieldVector< double, GridPart::dimension > ElmCenter = geo.center();

      std::vector< RangeType > PhiMonomialBasis( numBasisFct );
      std::vector< RangeType > Phi( numBasisFct );                    //

      // *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
      // *  *  *  *  *  *  *  *  *  *  assemble D-matrix *  *  *  *  *  *  *  *  *  *  *
      // assemble vertex dofs

      for( int i = 0; i < numElemVertices; ++i )
        if( agIndexSet.localIndex( ep, i, GridPart::dimension ) != -1 )
        {
          //          Dune::FieldVector<double, GridPart::dimension> GaussPoint = geo.corner(i);
          DomainType GlobalPoint = geo.corner( i );                              //
          //          Dune::FieldVector<double,GridPart::dimension> local =  ep->geometry().local(GaussPoint);
          DomainType LocalPoint = ep->geometry().local( GlobalPoint );
          basis.evaluateAll( LocalPoint, Phi );
          //          evalMonomialBasis(GlobalPoint,xE,hE,PhiMonomialBasis);
          // assemble D-matrix for vertex dofs
          int LocalDofInPolygon = agIndexSet.localIndex( ep, i, GridPart::dimension );                              //
          int GlobalVertexIndexofPolygon = agIndexSet.subIndex( ep, LocalDofInPolygon,  GridPart::dimension );

          //          int LocalVertexDofInPolygon =


          for( int jCol = 0; jCol < numBasisFct; ++jCol )
            Dlocal[ LocalDofInPolygon ][ jCol ] =  Phi[ jCol ];
        }

      // assemble edge dofs
      const IntersectionIteratorType isend = dfSpace.gridPart().iend( ep );                   // get the iterator-end for the current element
      for( IntersectionIteratorType it = dfSpace.gridPart().ibegin( ep ); it != isend; ++it )                   // looping over intersections
      {
        const IntersectionType &is = *it;
        int kdis_edge = ref.size( it->indexInInside(), 1, GridPart::dimension );                       // number of vertices on the edge
        const EntityPointerType InsideElement = is.inside();                         // pointer to outside element.
        Dune::GeometryType edgegt = it->type();
        int InsidePolygon = agglomeration.index( *InsideElement );                       // the polygon we are integrating
        int refface = it->indexInInside();                         // local face number based on the reference element class
        if( agIndexSet.localIndex( ep, refface, GridPart::dimension - 1 ) != -1 )
        {
          //std::cout << " ¬¬¬ " <<  is.geometry().center() << std::endl;
          int GaussLobattoOrder = POLORDER + 1;
          //          const Dune::QuadratureRule<typename GridPart::ctype, GridPart::dimension-1>& GaussLobattoRule =
          //              Dune::QuadratureRules<typename GridPart::ctype, GridPart::dimension-1>::
          //              rule(edgegt, GaussLobattoOrder, Dune::QuadratureType::GaussLobatto);
          typedef Dune::QuadratureRule< typename GridPart::ctype, GridPart::dimension - 1 > Quad;
          const Quad &quad = Dune::QuadratureRules< typename GridPart::ctype, GridPart::dimension - 1 >::rule( edgegt, GaussLobattoOrder, Dune::QuadratureType::GaussLobatto );
          typedef typename Dune::QuadratureRule< typename GridPart::ctype, GridPart::dimension - 1 >::iterator QuadIterator;
          QuadIterator qp = quad.begin();
          QuadIterator qend = quad.end();
          int ngauss = 1;
          for( QuadIterator igauss = qp + 2; igauss != qend; ++igauss )                            // this is actually your dof loop on a given edge.
          {
            //            std::cout << "==>> " << ngauss << " " <<igauss->position()[0] << std::endl;

            Dune::FieldVector< typename GridPart::ctype, GridPart::dimension > EdgeGaussPoint = is.geometry().global( igauss->position());

            /// physical location of the Gauss-Lobatto quadrature point
            DomainType GlobalPoint = is.geometry().global( igauss->position());
            // location in the reference bounding box for element T (note that this location is not in the reference bounding box yet!)
            DomainType LocalPoint = ep->geometry().local( GlobalPoint );
            // Evaluate the basis
            basis.evaluateAll( LocalPoint, Phi );
            fTemp << currentPolygon << " " << LeafElementIndex << " " << refface <<  " " << GlobalPoint << " , local = " << LocalPoint << std::endl;
            //                    std::cout << qp->weight() << std::endl;

//            int LocalDofInPolygon = agIndexSet.localIndex( ep, i, GridPart::dimension );//
//            int GlobalVertexIndexofPolygon = agIndexSet.subIndex( ep, LocalDofInPolygon,  GridPart::dimension );
            //            volume += qp->weight();
            ++ngauss;
          }
          // edge integration

          //
          //          for (auto iGL = GaussLobattoRule.begin(); iGL != GaussLobattoRule.end(); ++iGL) // Gauss-Lobattor quadrature loop
          //          {
          //
          //            std::cout << "-> " << iGL->position()[0] << std::endl;
          //            Dune::FieldVector<typename GridPart::ctype,GridPart::dimension > EdgeGaussPoint = is.geometry().global(iGL->position());
          //            std::cout << " igauss = "<< igauss << ", xi = "<< iGL->position()[0]  << std::endl;
          ////            std::cout << igauss << std::endl;
          //            ++igauss;
          //
          //          }



        }

      }
      // *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *

      // for H1 projection operator
      return 0;

      const IntersectionIteratorType iitend = dfSpace.gridPart().iend( ep );                   // get the iterator-end for the current element
      for( IntersectionIteratorType iit = dfSpace.gridPart().ibegin( ep ); iit != iitend; ++iit )                   // looping over intersections
      {
        const IntersectionType &intersection = *iit;
        int kdis_edge = ref.size( iit->indexInInside(), 1, GridPart::dimension );                       // number of vertices on the edge
        const EntityPointerType InsideElement = intersection.inside();                         // pointer to outside element.
        int InsidePolygon = agglomeration.index( *InsideElement );                       // the polygon we are integrating
        int refface = iit->indexInInside();                         // local face number based on the reference element class


        if( !intersection.boundary() )                            // (!intersection.boundary()
        {
          const EntityPointerType OutsideElement = intersection.outside();                               // pointer to outside element.
          int OutsidePolygon = agglomeration.index( *OutsideElement );                             // the polygon we are NOt integrating

          if( OutsidePolygon != InsidePolygon )
          {

            // lets get a local ids for edge
            int LocalEdgeId = agIndexSet.localIndex( ep, refface, GridPart::dimension-1 );
            int GlobalEdgeId = agIndexSet.subIndex( ep, LocalEdgeId,  GridPart::dimension-1 );
            std::cout << "loc/glo edge = " << refface << " "<< LocalEdgeId << " " << GlobalEdgeId << " "<< intersection.geometry().center() << std::endl;


            for( int i = 0; i < kdis_edge; ++i )
            {
              int localNodeNumber = ref.subEntity( refface, 1, i, GridPart::dimension );                                         // as per reference element class
              //              fpolyFaceInfo << currentPolygon << " " << LeafElementIndex << " " <<  refface << " " << geo.corner(localNodeNumber) << std::endl;
              int LocalDofInPolygon = agIndexSet.localIndex( ep, localNodeNumber, GridPart::dimension );

              fpolyFaceInfo << currentPolygon << " " << LeafElementIndex << " " << LocalPolygonalConnectivity[ LocalDofInPolygon ] <<  " " << refface << " " << geo.corner( localNodeNumber ) << std::endl;
              Dune::FieldVector< double, GridPart::dimension > unitOuterNormal = intersection.centerUnitOuterNormal();
              // Let's code the general case of obtained basis in P (k-1) later. For linear case, m0 = 1
              //              int LocalDofInPolygon = agIndexSet.localIndex( ep, localNodeNumber, GridPart::dimension );

              Dune::FieldVector< double, dimDomain > VertexCoordinate = geo.corner( localNodeNumber );

              evalDiffCoeff( VertexCoordinate, DiffusionCoefficent );
              evalReactionCoeff( VertexCoordinate, ReactionCoefficient );
              // I multiply by 1/2 below as we visit each vertex exactly twice while iterating over the polygon
              MeanReactionCoefficient = MeanReactionCoefficient + 0.5  * ReactionCoefficient;
              for( int jdim = 0; jdim < dimDomain; ++jdim )
                //We will evalulate the diffusion coefficient at each vertex
                MeanDiffusionCoefficient = MeanDiffusionCoefficient +  0.5 * DiffusionCoefficent [ jdim ][ jdim ] / ( dimDomain );


              for( int j = 0; j < GridPart::dimension; ++j )
                PI0X [ j ][ LocalDofInPolygon ] = PI0X[ j ][ LocalDofInPolygon ] + 0.5 * unitOuterNormal [ j ] * intersection.geometry().volume();

            }
          }

        }                         // (!intersection.boundary()


        if( intersection.boundary() )                            // this is always part of the polygon boundary as the intersection is on the boundary of the domain
          for( int i = 0; i < kdis_edge; ++i )
          {

            int localNodeNumber = ref.subEntity( refface, 1, i, GridPart::dimension );                                   // as per reference element class
            int LocalDofInPolygon = agIndexSet.localIndex( ep, localNodeNumber, GridPart::dimension );

            // lets get a local ids for edge
            int LocalEdgeId = agIndexSet.localIndex( ep, refface, GridPart::dimension-1 );
            int GlobalEdgeId = agIndexSet.subIndex( ep, LocalEdgeId,  GridPart::dimension-1 );
            std::cout << "loc/glo edge = " << refface << " "<< LocalEdgeId << " " << GlobalEdgeId << " "<< intersection.geometry().center() << std::endl;



            fpolyFaceInfo << currentPolygon << " " << LeafElementIndex << " " << LocalPolygonalConnectivity[ LocalDofInPolygon ] <<  " " << refface << " " << geo.corner( localNodeNumber ) << std::endl;
            Dune::FieldVector< double, GridPart::dimension > unitOuterNormal = intersection.centerUnitOuterNormal();
            //            int LocalDofInPolygon = agIndexSet.localIndex( ep, localNodeNumber, GridPart::dimension );

            Dune::FieldVector< double, dimDomain > VertexCoordinate = geo.corner( localNodeNumber );
            evalDiffCoeff( VertexCoordinate, DiffusionCoefficent );

            for( int jdim = 0; jdim < dimDomain; ++jdim )

              MeanDiffusionCoefficient = MeanDiffusionCoefficient +  0.5 * DiffusionCoefficent [ jdim ][ jdim ] / ( dimDomain );                                      // 0.5 because we will have two contributions from each vertex



            for( int j = 0; j < GridPart::dimension; ++j )
              PI0X [ j ][ LocalDofInPolygon ] = PI0X[ j ][ LocalDofInPolygon ] + 0.5 * unitOuterNormal [ j ] * intersection.geometry().volume();
          }


        // (!intersection.boundary()




      }                   // Segment for H1 projection operator ends

      // the finite element like integration
      // for linear case and for constant coefficient problems, we don't even need this integration provided the basis in (p-1) is a constant
      // We are still in the subelement loop



      const Dune::QuadratureRule< double, dimDomain > &Domainrule   \
        = Dune::QuadratureRules< double,  dimDomain >::rule( gt, 2 );

      int iGauss = 0;

      for( typename Dune::QuadratureRule< double, dimDomain >::const_iterator r
             = Domainrule.begin(); r != Domainrule.end(); ++r )
      {
        ++iGauss;
        // get the weight at the current quadrature point
        double weight = r->weight();

        double detjac = geo.integrationElement( r->position());

        dummyArea = dummyArea + weight * detjac;
        //        Dune::FieldVector<double, dimDomain> GaussPoint = geo.global(r->position());
        DomainType GlobalPoint = geo.global( r->position());

        //        Dune::FieldVector<double,GridPart::dimension> local =  ep->geometry().local(GaussPoint);
        DomainType LocalPoint = ep->geometry().local( GlobalPoint );

        basis.evaluateAll( LocalPoint, Phi );

        evalDiffCoeff( GlobalPoint, DiffusionCoefficent );
        evalReactionCoeff( GlobalPoint, ReactionCoefficient );
        evalTrueSolution( GlobalPoint, TrueSolution, TrueSolutionGradient );
        evalForcingFunction( GlobalPoint, TrueSolution, TrueSolutionGradient, ReactionCoefficient, forcingfunction );



        // Problem 1 - Poisson problem
        //        forcingfunction = M_PI * M_PI * sin(M_PI * GlobalPoint[0]) * GlobalPoint[1] * (1.0 - GlobalPoint[1]) + 2.0 * sin(M_PI * GlobalPoint[0]) ;
        // Problem 2 - Poisson problem
        // double forcingfunction = 2.0 * M_PI * M_PI * sin ( M_PI * GlobalPoint[0]) * sin ( M_PI * GlobalPoint [1]) ;
        // Problem 3 - Reaction-Diffusion equation
        // Problem 2 - Poisson problem

        for( int iRow = 0; iRow < nk; ++iRow )
        {
          F_mAlpha_local [ iRow ] += weight * detjac * Phi [ iRow ] * forcingfunction;
          if( iRow == 1 )
            fJunk1 << iGauss << weight << " " << detjac << " " << Phi[ iRow ] << " " << forcingfunction << std::endl;
        }

        fGaussPts << currentPolygon << " " << LeafElementIndex << " " << GlobalPoint << std::endl;



        // reaction matrix

        for( int iRow = 0; iRow < nk; ++iRow )
          for( int jCol = 0; jCol < nk; ++jCol )

            ReactionMatrix [ iRow ] [ jCol ] =  ReactionMatrix [ iRow ] [ jCol ]  + ReactionCoefficient * Phi[ iRow ] * Phi[ jCol ] * weight * detjac;

        AreaofDomain = AreaofDomain + detjac * weight;

        for( int iRow = 0; iRow < Kmat.rows(); ++iRow )
          for( int jCol = 0; jCol < Kmat.cols(); ++jCol )

            Kmat[ iRow ][ jCol ] = Kmat[ iRow ][ jCol ] +  weight * detjac * DiffusionCoefficent [ iRow ] [ jCol ] * Phi[ 0 ] *  Phi[ 0 ];                     // this is valid only for a linear case.

      }

    }             // finished iterating over sub-elements in a given polygon


    CenterOfMass_x = CenterOfMass_x / numSubElements;
    CenterOfMass_y = CenterOfMass_y / numSubElements;
    fpolyCoM << currentPolygon << " " << CenterOfMass_x << " " << CenterOfMass_y << std::endl;

    // matrix operations for each of the polygon

    // write out the D-matrix
    fDlocal << currentPolygon << std::endl;
    for( int iRow = 0; iRow < numLocalDof; ++iRow )
      for( int jCol = 0; jCol < nk; ++jCol )
        fDlocal << iRow << ", " << jCol << " , " << Dlocal[ iRow ][ jCol ] << std::endl;

    fDlocal << " " << std::endl;

    // get transpose of D
    for( int iRow = 0; iRow < nk; iRow++ )
      for( int jCol = 0; jCol < numLocalDof; jCol++ )
        DlocalTranspose[ iRow ][ jCol ] = Dlocal [ jCol ][ iRow ];

    // write out the DT-matrix
    fDlocalT << currentPolygon << std::endl;
    for( int iRow = 0; iRow < nk; ++iRow )
      for( int jCol = 0; jCol < numLocalDof; ++jCol )
        fDlocalT << iRow << ", " << jCol << " , " << DlocalTranspose[ iRow ][ jCol ] << std::endl;

    fDlocalT << " " << std::endl;

    // Do ( D^T D ): we store ( D^T D ) in DTDinversed directly as the size of DTD and DTDinverse is the same.
    for( int iRow = 0; iRow < nk; iRow++ )
      for( int jCol = 0; jCol < nk; jCol++ )

        for( int k = 0; k < numLocalDof; k++ )
          DTDinversed[ iRow ][ jCol ] = DTDinversed[ iRow ][ jCol ] + ( DlocalTranspose[ iRow ][ k ] * Dlocal[ k ][ jCol ] );


    // write out the DTD-matrix
    fDTD << currentPolygon << std::endl;
    for( int iRow = 0; iRow < nk; ++iRow )
      for( int jCol = 0; jCol < nk; ++jCol )
        fDTD << iRow << ", " << jCol << " , " << DTDinversed[ iRow ][ jCol ] << std::endl;

    fDTD << " " << std::endl;
    // do (D^T D)^-1
    DTDinversed.invert();             // reseult is in DTDinversed


    // write out the DTDI-matrix
    fDTDI << currentPolygon << std::endl;
    for( int iRow = 0; iRow < nk; ++iRow )
      for( int jCol = 0; jCol < nk; ++jCol )
        fDTDI << iRow << ", " << jCol << " , " << DTDinversed[ iRow ][ jCol ] << std::endl;

    fDTDI << " " << std::endl;


    for( int iRow = 0; iRow < nk; iRow++ )
      for( int jCol = 0; jCol < numLocalDof; jCol++ )

        for( int k = 0; k < nk; k++ )
          PI1[ iRow ][ jCol ] = PI1[ iRow ][ jCol ] + ( DTDinversed[ iRow ][ k ] * DlocalTranspose[ k ][ jCol ] );

    vec_mat_PI1.push_back( PI1 );

    fPI1 << currentPolygon << std::endl;
    for( int iRow = 0; iRow < nk; ++iRow )
      for( int jCol = 0; jCol < numLocalDof; ++jCol )
        fPI1 << iRow << ", " << jCol << " , " << PI1[ iRow ][ jCol ] << std::endl;

    fPI1 << " " << std::endl;

    for( int iRow = 0; iRow < numLocalDof; iRow++ )
      for( int jCol = 0; jCol < numLocalDof; jCol++ )

        for( int k = 0; k < nk; k++ )
          PI_PHI_1[ iRow ][ jCol ] = PI_PHI_1[ iRow ][ jCol ] + ( Dlocal[ iRow ][ k ] * PI1[ k ][ jCol ] );

    fPI_PHI_1 << currentPolygon << std::endl;
    for( int iRow = 0; iRow < numLocalDof; ++iRow )
      for( int jCol = 0; jCol < numLocalDof; ++jCol )
        fPI_PHI_1 << iRow << ", " << jCol << " , " << PI_PHI_1[ iRow ][ jCol ] << std::endl;

    fPI_PHI_1 << " " << std::endl;


    fPI0X << " ---  " << std::endl;
    fPI0X << currentPolygon << std::endl;
    for( int iRow = 0; iRow < GridPart::dimension; ++iRow )
      for( int jCol = 0; jCol < numLocalDof; ++jCol )
        fPI0X << iRow << ", " << jCol << " , " << PI0X[ iRow ][ jCol ] << std::endl;

    fPI0X << " ~ " << std::endl;


    KConsistencyLocal.resize( numLocalDof, numLocalDof );
    PI0Xt_x_Kmat.resize( numLocalDof, dimDomain );
    H0 = AreaofPolygon;
    PI0X /= H0;

    PI0XTransposed.resize( numLocalDof, dimDomain );

    // tranpose PI0X
    for( int iRow = 0; iRow < dimDomain; ++iRow )
      for( int jCol = 0; jCol < numLocalDof; ++jCol )

        PI0XTransposed[ jCol ][ iRow ] = PI0X[ iRow ][ jCol ];


    //  Do PI0X^T times Kmat
    for( int iRow = 0; iRow < numLocalDof; ++iRow )
      for( int jCol = 0; jCol < dimDomain; ++jCol )

        for( int k = 0; k < dimDomain; ++k )
          PI0Xt_x_Kmat[ iRow ][ jCol ] = PI0Xt_x_Kmat[ iRow ][ jCol ] + PI0XTransposed[ iRow ][ k ] * Kmat[ k ][ jCol ];

    // Do  ( PI0X^T times Kmat ) time ( PI0X)
    for( int iRow = 0; iRow < numLocalDof; ++iRow )
      for( int jCol = 0; jCol < numLocalDof; ++jCol )

        for( int k = 0; k < dimDomain; ++k )
          // remember, PI0X has already been assembled keeping in mind local dofs for the polygon, so while assembling the KConsistencyLocal, we directly do a matrix multiplication and
          // the resulting entries will be correctly placed at appropriate location in local matrix.
          KConsistencyLocal[ iRow ][ jCol ] = KConsistencyLocal[ iRow ][ jCol ] + PI0Xt_x_Kmat [ iRow ][ k ] * PI0X [ k ][ jCol ];

    for( int iRow = 0; iRow < numLocalDof; ++iRow )
      for( int jCol = 0; jCol < numLocalDof; ++jCol )
        fKclocal << iRow << ", " << jCol << " , " << KConsistencyLocal[ iRow ][ jCol ] << std::endl;





    KStbilityLocal.resize( numLocalDof, numLocalDof );           // stability

    Imatrix.resize( numLocalDof, numLocalDof );
    Imatrix = 0.0;
    for( int iRow = 0; iRow < Imatrix.rows(); ++iRow )
      Imatrix[ iRow ][ iRow ] = 1.0;

    // first store transpose of (I-PI_PHI_1) in KStbilityLocal
    //  for (int iRow = 0; iRow < numPolygonVertices; ++iRow) {
    //    for (int jCol = 0; jCol < numPolygonVertices; ++jCol) {
    //      KStbilityLocal [iRow][jCol] = Imatrix [jCol][iRow] - PI_PHI_1[jCol][iRow];
    //    }
    //  }


    I_minus_PI_PHI_1.resize( numLocalDof, numLocalDof );
    I_minus_PI_PHI_1_Transposed.resize( numLocalDof, numLocalDof );


    for( int iRow = 0; iRow < numLocalDof; ++iRow )
      for( int jCol = 0; jCol < numLocalDof; ++jCol )
        I_minus_PI_PHI_1 [ iRow ][ jCol ] =  Imatrix [ iRow ][ jCol ]- PI_PHI_1 [ iRow ][ jCol ];

    for( int iRow = 0; iRow < numLocalDof; ++iRow )
      for( int jCol = 0; jCol < numLocalDof; ++jCol )
        I_minus_PI_PHI_1_Transposed [ iRow ][ jCol ] = I_minus_PI_PHI_1 [ jCol ][ iRow ];

    KStbilityLocal = 0;             // initialise
    // first store I_minus_PI_PHI_1_Transposed in KStbilityLocal
    for( int iRow = 0; iRow < numLocalDof; ++iRow )
      for( int jCol = 0; jCol < numLocalDof; ++jCol )
        KStbilityLocal [ iRow ][ jCol ] = I_minus_PI_PHI_1_Transposed [ iRow ][ jCol ];


    double OneOverDimDomain = 1.0 / dimDomain;
    double LocalMeshSize = std::pow( AreaofPolygon, OneOverDimDomain );
    AverageMeshSize += LocalMeshSize;
    MeanDiffusionCoefficient /= numLocalDof;
    MeanReactionCoefficient /= numLocalDof;

    StabilityScalingFactor = MeanDiffusionCoefficient * ( pow( LocalMeshSize, GridPart::dimension - 2 ) );
    StabilityScalingFactor += MeanReactionCoefficient * ( pow( LocalMeshSize, GridPart::dimension ));
    KStbilityLocal.rightmultiply( I_minus_PI_PHI_1 );           // In order to do [I_minus_PI_PHI_1_Transposed]*[I_minus_PI_PHI_1], you write [I_minus_PI_PHI_1].leftmultiply[I_minus_PI_PHI_1_Transposed] i.e. leftmultiply I_minus_PI_PHI_1 by I_minus_PI_PHI_1_Transposed.
    KStbilityLocal *= StabilityScalingFactor;

    KStiffLocal.resize( numLocalDof, numLocalDof );
    KStiffLocal += KConsistencyLocal;
    KStiffLocal += KStbilityLocal;



    for( int iRow = 0; iRow < nk; ++iRow )
      for( int jCol = 0; jCol < numLocalDof; ++jCol )
        PI1T [ jCol ][ iRow ] = PI1 [ iRow ][ jCol ];


    f_load_local = 0;
    for( int iRow = 0; iRow < numLocalDof; iRow++ )

      for( int k = 0; k < nk; ++k )
        f_load_local[ iRow ] = f_load_local[ iRow ]
                               + ( PI1T[ iRow ][ k ] * F_mAlpha_local[ k ] );



    // reaction terms
    PI1T_x_ReactionMatrix.resize( numLocalDof, nk );
    PI1T_x_ReactionMatrix = 0;
    // Do  ( PI1^T times ReactionMatrix )
    for( int iRow = 0; iRow < numLocalDof; ++iRow )
      for( int jCol = 0; jCol < nk; ++jCol )

        for( int k = 0; k < nk; ++k )

          PI1T_x_ReactionMatrix [ iRow ][ jCol ] = PI1T_x_ReactionMatrix[ iRow ][ jCol ] + PI1T [ iRow ][ jCol ] * ReactionMatrix [ jCol ][ k ];

    // assemble mass matrix
    for( int iRow = 0; iRow < numLocalDof; ++iRow )
      for( int jCol = 0; jCol < numLocalDof; ++jCol )

        for( int k = 0; k < nk; ++k )

          MassLocal [ iRow ][ jCol ] = MassLocal[ iRow ][ jCol ] + PI1T_x_ReactionMatrix [ iRow ][ k ] * PI1 [ k ][ jCol ];



    // assembly of global stiffness matrix





    for( int iRow = 0; iRow < numLocalDof; ++iRow )
    {
      int iGlobalRow = LocalPolygonalConnectivity [ iRow ];
      for( int jCol = 0; jCol < numLocalDof; ++jCol )
      {
        int jGlobalCol = LocalPolygonalConnectivity [ jCol ];
        KStiffGlobal [ iGlobalRow ][ jGlobalCol ] += KStiffLocal [ iRow ][ jCol ] + MassLocal [ iRow ][ jCol ];
      }
      f_load_global[ iGlobalRow ] += f_load_local [ iRow ];
    }





    //    AreaofDomain = AreaofDomain + AreaofPolygon ;
    ++currentPolygon;

  }       // Polygon Loop


  // solution proc
  b_rhs_global.resize( numVtxPolygonalMesh );
  b_rhs_global = f_load_global;
  A_lhs_global.resize( numVtxPolygonalMesh, numVtxPolygonalMesh );
  A_lhs_global_inverse.resize( numVtxPolygonalMesh, numVtxPolygonalMesh );
  A_lhs_global = KStiffGlobal;
  AverageMeshSize /= agglomeration.size();

  // set Dirichlet boundary conditions:
  // replace lines in A_lhs_global_inverse related to Dirichlet vertices by trivial lines
  int numDirichletEdges = 0;


  for( IteratorType it = dfSpace.begin(); it != end; ++it )
  {
    const EntityType &entity = *it;
    Dune::GeometryType gt = entity->type();
    auto &ref = Dune::ReferenceElements< double, dimDomain >::general( gt );
    typedef typename ElementIterator::Entity::Geometry LeafGeometry;
    const LeafGeometry geo = entity->geometry();
    const IntersectionIteratorType endiit = gridPart.iend( entity );

    for( IntersectionIteratorType iit = gridPart.ibegin( entity );
         iit != endiit; ++iit )
    {
      const IntersectionType &intersection = *iit;
      int refface = iit->indexInInside();                   // local face number based on the reference element class
      if( intersection.boundary())
      {

        int kdis_edge = ref.size( iit->indexInInside(), 1, GridPart::dimension );                       // number of vertices on the edge

        for( int i = 0; i < kdis_edge; ++i )
        {

          int localNodeNumber = ref.subEntity( refface, 1, i, GridPart::dimension );                             // as per reference element class

          int LocalDofInPolygon = agIndexSet.localIndex( entity, localNodeNumber, GridPart::dimension );

          int GlobalVertexIndexofPolygon = agIndexSet.subIndex( entity, LocalDofInPolygon,  GridPart::dimension );


          A_lhs_global[ GlobalVertexIndexofPolygon ] = 0.0;
          A_lhs_global[ GlobalVertexIndexofPolygon ][ GlobalVertexIndexofPolygon ] = 1.0;
          b_rhs_global[ GlobalVertexIndexofPolygon ] = 0.0;


        }

        ++numDirichletEdges;
      }
    }
  }



  // solve linear system

  x_soln_global.resize( b_rhs_global.N(), false );
  x_soln_global = 2.0;

  A_lhs_global_inverse = 0;
  A_lhs_global_inverse = A_lhs_global;


  for( int iRow = 0; iRow < numVtxPolygonalMesh; ++iRow )
    for( int jCol = 0; jCol < numVtxPolygonalMesh; ++jCol )
      fAmatrix << " " << iRow << " " << jCol << " " << A_lhs_global [ iRow ] [ jCol ] << std::endl;


  A_lhs_global_inverse.invert();


  double fro_norm = A_lhs_global.frobenius_norm();
  double fro_norm_iA = A_lhs_global_inverse.frobenius_norm();
  double cond_A = fro_norm * fro_norm_iA;



  //  //
  A_lhs_global.solve( x_soln_global, b_rhs_global );

  for( int iRow = 0; iRow < numVtxPolygonalMesh; ++iRow )
    fXVemSoln << iRow <<" " <<PolygonalMeshVertexIDs [ iRow ] << " " << x_soln_global [ iRow ] << std::endl;


  // x_soln_global now contains the solution in VEM space. Let us project it back into the polynomial space
  currentPolygon = 0;
  oldPolygon = -1;

  double VemL2error = 0;
  double VemH1error = 0;
  // Local solution vectors
  ScalarField LocalSolutionVector;
  ScalarField L2projectedVemSoln_local;       // this should be PiStarDelta * u_local, where u_local is the local solution vector corresponding to the element.

  for( auto PolygonIterator = PolygonalMeshIDs.begin(); PolygonIterator != PolygonalMeshIDs.end(); ++PolygonIterator )
  {


    int numPolygonVertices = NVertexVector[ currentPolygon ];
    numLocalDof = POLORDER * numPolygonVertices + numTermsin_k_minus_2;
    LocalSolutionVector.resize( numLocalDof );

    L2projectedVemSoln_local.resize( nk );
    LocalPolygonalConnectivity = PolygonalMeshConnectivityArray[ currentPolygon ];

    // ==========================================================================================
    // loop for mapping the global solution to local dof in the polygon. This is an extra loop but I can't
    // see how this can be avoided. Note that the vector "LocalPolygonalConnectivity" has the Global vertex numbers
    // of the current polygon in ascending order and we don't know yet which global vertex corresponds to which local
    // vertex of the polygon. This information can only be obtained using localIndex function as below.

    for( auto it = ( *PolygonIterator ).begin(); it != ( *PolygonIterator ).end(); ++it )
    {
      // Define the Leaf Geometry
      typedef typename ElementIterator::Entity::Geometry LeafGeometry;
      // get the element pointer
      const ElementPointer ep = gridPart.grid().entityPointer( *it );
      // get the geometry of the element
      const LeafGeometry geo = ep->geometry();
      // find out how many vertices there are in the underlying element T
      int numElemVertices = geo.corners();
      // Now, find out the local vertex numbers for current polygon for a given local vertex in T
      for( int iRow = 0; iRow < numElemVertices; ++iRow )
      {

        // find what is the local dof number (LocalDofInPolygon) in the polygon for a given local dof in T (iRow)
        int LocalDofInPolygon = agIndexSet.localIndex( ep, iRow, GridPart::dimension );
        // Remember, LocalDofInPolygon can have -1 value meaning it is internal vertex and we need to skip it
        if( LocalDofInPolygon != -1 )
        {
          // find out what is the Global vertex number in the connectivity vector for a given local dof number
          int GlobalDofinPolygonalMesh = LocalPolygonalConnectivity[ LocalDofInPolygon ];
          // store the solution at GlobalDofinPolygonalMesh in global solution vector at
          // the local solution vector and at corresponding local dof in the polygon
          LocalSolutionVector[ LocalDofInPolygon ] = x_soln_global [ GlobalDofinPolygonalMesh ];
        }
      }
    }

    PI1 = vec_mat_PI1 [ currentPolygon ];
    L2projectedVemSoln_local = 0;
    for( int iRow = 0; iRow < nk; iRow++ )
      for( int jCol = 0; jCol < numPolygonVertices; jCol++ )

        L2projectedVemSoln_local[ iRow ] = L2projectedVemSoln_local[ iRow ] + ( PI1[ iRow ][ jCol ] * LocalSolutionVector[ jCol ] );

    // ==========================================================================================
    // L2projectedVemSoln_local is in Pk polynomial space of degree k
    // Let us do the error calculations now.
    int elementsInThisPolygon = 0;

    for( auto it = ( *PolygonIterator ).begin(); it != ( *PolygonIterator ).end(); ++it )
    {

      ++elementsInThisPolygon;

      typedef typename ElementIterator::Entity::Geometry LeafGeometry;

      const ElementPointer ep = gridPart.grid().entityPointer( *it );

      const LeafGeometry geo = ep->geometry();

      Dune::GeometryType gt = ep->type();

      auto &ref = Dune::ReferenceElements< double, dimDomain >::general( gt );

      const int LeafElementIndex = lset.index( ep );                 // Global element number of the current element

      auto &basis = dfSpace.basisFunctionSet( ep );

      const int numBasisFct = basis.size();

      int numElemVertices = geo.corners();

      // Quadrature loop for error calculations

      int iGauss = 0;

      std::vector< RangeType > Phi( numBasisFct );                    // basis functions

      std::vector< JacobianRangeType > dPhi( numBasisFct );                  // derivatives of basis functions


      const Dune::QuadratureRule< double, dimDomain > &Domainrule   \
        = Dune::QuadratureRules< double,  dimDomain >::rule( gt, 8 );

      for( typename Dune::QuadratureRule< double, dimDomain >::const_iterator r
             = Domainrule.begin(); r != Domainrule.end(); ++r )
      {
        ++iGauss;

        // get the weight at the current quadrature point
        double weight = r->weight();

        double detjac = geo.integrationElement( r->position());

        DomainType GlobalPoint = geo.global( r->position());

        //        Dune::FieldVector<double,GridPart::dimension> local =  ep->geometry().local(GaussPoint);
        DomainType LocalPoint = ep->geometry().local( GlobalPoint );

        basis.evaluateAll( LocalPoint, Phi );
        basis.jacobianAll( LocalPoint, dPhi );

        RangeType TrueSolution;
        RangeType ApproxVemSoln;

        JacobianRangeType TrueSolutionGradient;
        JacobianRangeType ApproxVemSolnGradient;

        evalTrueSolution( GlobalPoint, TrueSolution, TrueSolutionGradient );

        for( int iRow = 0; iRow < nk; iRow++ )
        {

          Dune::FieldMatrix< double, 1, GridPart::dimension > gradientValue = dPhi [ iRow ];

          ApproxVemSoln = ApproxVemSoln + Phi[ iRow ] * L2projectedVemSoln_local[ iRow ];

          for( int idim = 0; idim < GridPart::dimension; ++idim )
            ApproxVemSolnGradient [ 0 ][ idim ] = ApproxVemSolnGradient [ 0 ][ idim ] + gradientValue [ 0 ][ idim ] *  L2projectedVemSoln_local[ iRow ];

        }

        VemL2error = VemL2error + ( TrueSolution - ApproxVemSoln ) * ( TrueSolution - ApproxVemSoln ) * weight * detjac;

        for( int idim = 0; idim < GridPart::dimension; ++idim )
        {
          double differenceInGradient = TrueSolutionGradient[ 0 ] [ idim ] - ApproxVemSolnGradient [ 0 ][ idim ];
          VemH1error = VemH1error +  ( differenceInGradient * differenceInGradient ) * weight * detjac;
        }
      }

    }

    ++currentPolygon;
  }

  VemL2error = sqrt( VemL2error );
  VemH1error = sqrt( VemH1error );

  std::cout << "Ndof = " << numVtxPolygonalMesh << std::endl;
  std::cout << "L2 error = " << VemL2error << std::endl;
  std::cout << "H1 error = " << VemH1error << std::endl;
  std::cout << VemL2error << " " << VemH1error << std::endl;
  //  fOut << "NDof " << "h " << "L2 error " << "H1 error " <<  "cond. no. " << std::endl;
  fOut << numVtxPolygonalMesh << " " << AverageMeshSize << " " << VemL2error << " " << VemH1error << " " << cond_A << std::endl;
  std::cout << "done in algo" << std::endl;
  return error;
} // double algorithm function



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

  const auto sectionMap = Gmsh::readFile( "/home/gcd3/project/dune-vem/data/partitioned-mesh.msh" );
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
  std::transform( agglomerateIndices.begin(), agglomerateIndices.end(), agglomerateIndices.begin(), [ ] ( int i ) {
    return i-1;
  } );

  // create grid part and agglomeration

  typedef Dune::Fem::LeafGridPart< Grid > GridPart;
  GridPart gridPart( *grid );

  algorithm( gridPart, agglomerateIndices );

  // write the grid you originally imported
  typedef Grid::LeafGridView GV;
  Dune::GmshWriter< typename Grid::LeafGridView > writer( grid->leafGridView());
  writer.write( "../../output/yourmesh.msh" );

  std::cout << "Done..!" << std::endl;
  return 0;
}
catch( const Dune::Exception &e )
{
  std::cout << e << std::endl;
  return 1;
}
