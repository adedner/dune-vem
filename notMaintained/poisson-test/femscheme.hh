/**************************************************************************

  The dune-fem module is a module of DUNE (see www.dune-project.org).
  It is based on the dune-grid interface library
  extending the grid interface by a number of discretization algorithms
  for solving non-linear systems of partial differential equations.

  Copyright (C) 2003 - 2014 Robert Kloefkorn
  Copyright (C) 2003 - 2010 Mario Ohlberger
  Copyright (C) 2004 - 2014 Andreas Dedner
  Copyright (C) 2005        Adrian Burri
  Copyright (C) 2005 - 2014 Mirko Kraenkel
  Copyright (C) 2006 - 2014 Christoph Gersbacher
  Copyright (C) 2006 - 2014 Martin Nolte
  Copyright (C) 2011 - 2014 Tobias Malkmus
  Copyright (C) 2012 - 2014 Stefan Girke
  Copyright (C) 2013 - 2014 Claus-Justus Heine
  Copyright (C) 2013 - 2014 Janick Gerstenberger
  Copyright (C) 2013        Sven Kaulman
  Copyright (C) 2013        Tom Ranner


  The dune-fem module is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of
  the License, or (at your option) any later version.

  The dune-fem module is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along
  with this program; if not, write to the Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

 **************************************************************************/
#ifndef ELLIPT_FEMSCHEME_HH
#define ELLIPT_FEMSCHEME_HH

// iostream includes
#include <iostream>

// include discrete function space
#include <dune/fem/space/discontinuousgalerkin.hh>
#include <dune/vem/agglomeration/agglomeration.hh>
#include <dune/vem/agglomeration/dgspace.hh>

// adaptation ...
#include <dune/fem/function/adaptivefunction.hh>
#include <dune/fem/space/common/adaptmanager.hh>

// include discrete function
#include <dune/fem/function/blockvectorfunction.hh>

// include linear operators
#include <dune/fem/operator/linear/spoperator.hh>
#include <dune/fem/solver/diagonalpreconditioner.hh>

#include <dune/fem/operator/linear/istloperator.hh>
#include <dune/fem/solver/istlsolver.hh>
#include <dune/fem/solver/cginverseoperator.hh>

/*********************************************************/

// include norms
#include <dune/fem/misc/l2norm.hh>
#include <dune/fem/misc/h1norm.hh>

// include parameter handling
#include <dune/fem/io/parameter.hh>

// local includes
#include "probleminterface.hh"

#include "model.hh"

#include "dgelliptic.hh"
#include "dgrhs.hh"



// ISTL is only working of LinearOperators with matrix representation
#if HAVE_DUNE_ISTL && WANT_ISTL
#define USE_ISTL 1
#endif

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

// FemScheme
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
class FemScheme
{
public:
  //! type of the mathematical model
  typedef Model ModelType ;

  //! grid view (e.g. leaf grid view) provided in the template argument list
  typedef typename ModelType::GridPartType GridPartType;

  //! type of underyling hierarchical grid needed for data output
  typedef typename GridPartType::GridType GridType;




  //! type of function space (scalar functions, \f$ f: \Omega -> R \f$)
  typedef typename ModelType :: FunctionSpaceType   FunctionSpaceType;

  //! choose type of discrete function space
  typedef Dune::Vem::Agglomeration <GridPartType>  AgglomerationType;
//  typedef Dune::Fem::DiscontinuousGalerkinSpace< FunctionSpaceType, GridPartType, POLORDER > DiscreteFunctionSpaceType;
  typedef Dune::Vem::AgglomerationDGSpace < FunctionSpaceType, GridPartType, POLORDER > DiscreteFunctionSpaceType;



  // choose type of discrete function, Matrix implementation and solver implementation
#if USE_ISTL
  typedef Dune::Fem::ISTLBlockVectorDiscreteFunction< DiscreteFunctionSpaceType > DiscreteFunctionType;
  typedef Dune::Fem::ISTLLinearOperator< DiscreteFunctionType, DiscreteFunctionType > LinearOperatorType;
  typedef Dune::Fem::ISTLCGOp< DiscreteFunctionType, LinearOperatorType > LinearInverseOperatorType;
#else
  typedef Dune::Fem::AdaptiveDiscreteFunction< DiscreteFunctionSpaceType > DiscreteFunctionType;
  typedef Dune::Fem::SparseRowLinearOperator< DiscreteFunctionType, DiscreteFunctionType > LinearOperatorType;
  typedef Dune::Fem::CGInverseOperator< DiscreteFunctionType > LinearInverseOperatorType;
#endif

  /*********************************************************/

  //! define Laplace operator
  typedef DifferentiableDGEllipticOperator< LinearOperatorType, ModelType > EllipticOperatorType;




  FemScheme( GridPartType &gridPart,
      const ModelType& implicitModel,
      AgglomerationType& agglomeration)
  : implicitModel_( implicitModel ),
    gridPart_( gridPart ),
    agglomeration_(agglomeration),
    discreteSpace_( agglomeration),
    solution_( "solution", discreteSpace_ ),
    rhs_( "rhs", discreteSpace_ ),
    // the elliptic operator (implicit)
    implicitOperator_( implicitModel_, discreteSpace_ ),
    // create linear operator (domainSpace,rangeSpace)
    linearOperator_( "assempled elliptic operator", discreteSpace_, discreteSpace_ ),
    // exact solution
    solverEps_( Dune::Fem::Parameter::getValue< double >( "poisson.solvereps", 1e-8 ) )
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
    // assemble rhs
    assembleDGRHS ( implicitModel_, rhs_ );
  }

  //! solve the system - bool parameter
  //! false: only assemble if grid has changed
  //! true:  assemble in any case
  void solve ( bool assemble )
  {
    if( assemble )
    {
      // assemble linear operator (i.e. setup matrix)
      implicitOperator_.jacobian( solution_ , linearOperator_ );
    }

    // inverse operator using linear operator
    LinearInverseOperatorType invOp( linearOperator_, solverEps_, solverEps_ );
    // solve system
    invOp( rhs_, solution_ );
  }

protected:
  const ModelType& implicitModel_;   // the mathematical model

  GridPartType  &gridPart_;         // grid part(view), e.g. here the leaf grid the discrete space is build with

  DiscreteFunctionSpaceType discreteSpace_; // discrete function space
  DiscreteFunctionType solution_;   // the unknown
  DiscreteFunctionType rhs_;        // the right hand side

  EllipticOperatorType implicitOperator_; // the implicit operator

  LinearOperatorType linearOperator_;  // the linear operator (i.e. jacobian of the implicit)

  AgglomerationType agglomeration_;

  const double solverEps_ ; // eps for linear solver
};

#endif // end #if ELLIPT_FEMSCHEME_HH
