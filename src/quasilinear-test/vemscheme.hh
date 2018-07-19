#ifndef ELLIPT_VEMSCHEME_HH
#define ELLIPT_VEMSCHEME_HH
/*
 * 1) this scheme file is copied from /dune-fem-howto/05.nonlinear/source/
 *
 * */
// iostream includes
#include <iostream>

// include discrete function space
//#include <dune/fem/space/lagrange.hh>
#include <dune/fem/space/discontinuousgalerkin.hh>
#include <dune/vem/space/agglomeration.hh>

// adaptation ...
#include <dune/fem/function/adaptivefunction.hh>
#include <dune/fem/space/common/adaptmanager.hh>

// include discrete function
#include <dune/fem/function/blockvectorfunction.hh>

// include linear operators
#include <dune/fem/operator/linear/spoperator.hh>
#include <dune/fem/solver/diagonalpreconditioner.hh>
//
#include <dune/fem/operator/linear/istloperator.hh>
#include <dune/fem/solver/istlsolver.hh>
#include <dune/fem/solver/oemsolver.hh>
#include <dune/fem/solver/cginverseoperator.hh>
#include <dune/fem/solver/newtoninverseoperator.hh>
#include <dune/fem/solver/umfpacksolver.hh>

/*********************************************************/

// include norms
#include <dune/fem/misc/l2norm.hh>
#include <dune/fem/misc/h1norm.hh>

// include parameter handling
#include <dune/fem/io/parameter.hh>

// local includes
#include "rhs.hh"
#include "vemelliptic.hh"

// DataOutputParameters
// --------------------

struct DataOutputParameters
: public Dune::Fem::LocalParameter< Dune::Fem::DataOutputParameters, DataOutputParameters >
{
  DataOutputParameters ( const int step )
  : step_( step )
  {}

  DataOutputParameters ( const DataOutputParameters &other )
  : step_( other.step_ )
  {}

  std::string prefix () const
  {
    std::stringstream s;
    s << "poisson-" << step_ << "-";
    return s.str();
  }

private:
  int step_;
};

// VemScheme
//----------

/*******************************************************************************
 * template arguments are:
 * - GridPsrt: the part of the grid used to tesselate the
 *             computational domain
 * - Model: description of the data functions and methods required for the
 *          elliptic operator (massFlux, diffusionFlux)
 *     Model::ProblemType boundary data, exact solution,
 *                        and the type of the function space
 *******************************************************************************/
template < class Model >
class VemScheme
{
public:
  //! type of the mathematical model
  typedef Model ModelType ;

  //! grid view (e.g. leaf grid view) provided in the template argument list
  typedef typename ModelType::GridPartType GridPartType;

  //! type of underyling hierarchical grid needed for data output
  typedef typename GridPartType::GridType GridType;

  //! type of function space (scalar functions, \f$ f: \Omega -> R \f$)
  typedef typename ModelType::FunctionSpaceType   FunctionSpaceType;

  //! choose type of discrete function space
  //typedef Dune::Fem::LagrangeDiscreteFunctionSpace< FunctionSpaceType, GridPartType, POLORDER > DiscreteFunctionSpaceType;
  typedef Dune::Vem::Agglomeration <GridPartType>  AgglomerationType;
  typedef Dune::Vem::AgglomerationIndexSet< GridPartType > AgglomerationIndexSetType;
  typedef Dune::Vem::AgglomerationVEMSpace< FunctionSpaceType, GridPartType, POLORDER > VemSpaceType;

  // choose type of discrete function, Matrix implementation and solver implementation
#if HAVE_DUNE_ISTL && WANT_ISTL
  typedef Dune::Fem::ISTLBlockVectorDiscreteFunction< VemSpaceType > DiscreteFunctionType;
  typedef Dune::Fem::ISTLLinearOperator< DiscreteFunctionType, DiscreteFunctionType > LinearOperatorType;
  //typedef Dune::Fem::ISTLBICGSTABOp< DiscreteFunctionType, LinearOperatorType > LinearInverseOperatorType;
  //typedef Dune::Fem::ISTLGMResOp< DiscreteFunctionType, LinearOperatorType > LinearInverseOperatorType;
  typedef Dune::Fem::ISTLCGOp< DiscreteFunctionType, LinearOperatorType > LinearInverseOperatorType;
#else
  typedef Dune::Fem::AdaptiveDiscreteFunction< VemSpaceType > DiscreteFunctionType;
  typedef Dune::Fem::SparseRowLinearOperator< DiscreteFunctionType, DiscreteFunctionType > LinearOperatorType;
  //typedef Dune::Fem::CGInverseOperator< DiscreteFunctionType > LinearInverseOperatorType;
  //typedef Dune::Fem::OEMBICGSTABOp< DiscreteFunctionType, LinearOperatorType > LinearInverseOperatorType;
  typedef Dune::Fem::UMFPACKOp< DiscreteFunctionType, LinearOperatorType > LinearInverseOperatorType;
#endif

  /*********************************************************/

  //! define Laplace operator
  typedef DifferentiableVEMEllipticOperator < LinearOperatorType, ModelType > EllipticOperatorType;
  //! [Newton solver]
  typedef Dune::Fem::NewtonInverseOperator< LinearOperatorType, LinearInverseOperatorType > InverseOperatorType;
  //! [Newton solver]

  VemScheme( GridPartType &gridPart, const ModelType &implicitModel, AgglomerationType& agglomeration  )
  : implicitModel_( implicitModel ),
    gridPart_( gridPart ),
    indexSet_( agglomeration ),
    discreteSpace_( agglomeration ),
    solution_( "solution", discreteSpace_ ),
    rhs_( "rhs", discreteSpace_ ),
    // the elliptic operator (implicit)
    implicitOperator_( implicitModel_, discreteSpace_ )
  {
    // set all DoF to zero
    solution_.clear();
  }

  const DiscreteFunctionType &solution() const
  {
    return solution_;
  }

  //! sotup the right hand side
  void prepare()
  {
    // set boundary values for solution
    implicitOperator_.prepare( implicitModel_.dirichletBoundary(), solution_ );

    // assemble rhs
    assembleRHS ( implicitModel_, implicitModel_.rightHandSide(), implicitModel_.neumanBoundary(), rhs_ );

    // apply constraints, e.g. Dirichlet contraints, to the result
    implicitOperator_.prepare( solution_, rhs_ );
  }

  //! solve the system
  void solve ( bool assemble )
  {
    InverseOperatorType invOp( implicitOperator_ );
    invOp( rhs_, solution_ );
  }

protected:
  const ModelType& implicitModel_;   // the mathematical model

  GridPartType  &gridPart_;         // grid part(view), e.g. here the leaf grid the discrete space is build with
  AgglomerationIndexSetType indexSet_;
  //DiscreteFunctionSpaceType discreteSpace_; // discrete function space
  VemSpaceType discreteSpace_; // discrete function space
  DiscreteFunctionType solution_;   // the unknown
  DiscreteFunctionType rhs_;        // the right hand side

  EllipticOperatorType implicitOperator_; // the implicit operator
};

#endif // end #if ELLIPT_VEMSCHEME_HH
