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
#ifndef DGRHS_HH
#define DGRHS_HH

#include <dune/fem/quadrature/cachingquadrature.hh>

#include "rhs.hh"


// assembleRHS
// -----------

template< class Model, class DiscreteFunction >
void assembleDGRHS ( const Model &model, DiscreteFunction &rhs )
{
  assembleRHS( model.rightHandSide(), rhs );
  if ( ! model.hasDirichletBoundary() )
  {
    std::cout <<"problem has dirichlet boundary, exiting.." << std::endl;
    return;
  }

  typedef typename DiscreteFunction::DiscreteFunctionSpaceType DiscreteFunctionSpaceType;
  typedef typename DiscreteFunction::LocalFunctionType LocalFunctionType;

  typedef typename DiscreteFunctionSpaceType::IteratorType IteratorType;
  typedef typename IteratorType::Entity EntityType;
  typedef typename EntityType::Geometry GeometryType;
  typedef typename LocalFunctionType::RangeType RangeType;
  typedef typename LocalFunctionType::JacobianRangeType JacobianRangeType;
  typedef typename DiscreteFunctionSpaceType::DomainType DomainType;
  static const int dimDomain = LocalFunctionType::dimDomain;
  static const int dimRange = LocalFunctionType::dimRange;

  typedef typename DiscreteFunctionSpaceType::GridPartType GridPartType;
  typedef typename GridPartType::IntersectionIteratorType IntersectionIteratorType;
  typedef typename IntersectionIteratorType::Intersection IntersectionType;

  typedef Dune::Fem::ElementQuadrature< GridPartType, 1 > FaceQuadratureType;

  const DiscreteFunctionSpaceType &dfSpace = rhs.space();
  const int quadOrder = 2*dfSpace.order()+1;

  std::ofstream frhs ("rhs-orig.dat");
  std::ofstream fdump1 ("boundaryIntegrals-orig.dat");

  const IteratorType end = dfSpace.end();
  for( IteratorType it = dfSpace.begin(); it != end; ++it )
  {
    const EntityType &entity = *it;
    if ( !entity.hasBoundaryIntersections() )
      continue;

    const GeometryType &geometry = entity.geometry();
    double area = geometry.volume();

    std::cout << "visiting elm c(.) = " << geometry.center() << std::endl;


    LocalFunctionType rhsLocal = rhs.localFunction( entity );

    const IntersectionIteratorType iitend = dfSpace.gridPart().iend( entity );
    for( IntersectionIteratorType iit = dfSpace.gridPart().ibegin( entity ); iit != iitend; ++iit ) // looping over intersections
    {
      const IntersectionType &intersection = *iit;
      if ( ! intersection.boundary() ) // i.e. if intersection is on boundary: nothing to be done for Neumann zero b.c.
        continue;                      // since [u] = 0  and grad u.n = 0
      Dune::FieldVector< int, RangeType::dimension > components( 1 );
      if ( ! model.isDirichletIntersection( intersection, components ) )
        continue;

      typedef typename IntersectionType::Geometry  IntersectionGeometryType;
      const IntersectionGeometryType &intersectionGeometry = intersection.geometry();

      const double intersectionArea = intersectionGeometry.volume();
      const double beta = model.penalty() * intersectionArea / area;

      FaceQuadratureType quadInside( dfSpace.gridPart(), intersection, quadOrder, FaceQuadratureType::INSIDE );
      const size_t numQuadraturePoints = quadInside.nop();
      for( size_t pt = 0; pt < numQuadraturePoints; ++pt )
      {
        const typename FaceQuadratureType::LocalCoordinateType &x = quadInside.localPoint( pt );
        const DomainType normal = intersection.integrationOuterNormal( x );
        fdump1 << geometry.center() << " " << intersectionGeometry.center() << " " << x << " " << normal << std::endl;
        std::cout << geometry.center() << " " << intersectionGeometry.center() << " " << x << " " << normal << std::endl;
        const double weight = quadInside.weight( pt );

        RangeType value;
        JacobianRangeType dvalue,advalue;

        RangeType vuOut;
        model.g( RangeType(0), entity, quadInside.point(pt), vuOut );

        value = vuOut;
        value *= beta * intersectionGeometry.integrationElement( x );

        //  [ u ] * { grad phi_en } = -normal(u+ - u-) * 0.5 grad phi_en
        // here we need a diadic product of u x n
        for (int r=0;r<dimRange;++r)
          for (int d=0;d<dimDomain;++d)
            dvalue[r][d] = -0.5 * normal[d] * vuOut[r];

        model.init(entity);
        model.diffusiveFlux( quadInside[ pt ], vuOut, dvalue, advalue );

        value *= weight;
        advalue *= weight;
        rhsLocal.axpy( quadInside[ pt ], value, advalue );
      }
    }
  }
  rhs.print(frhs);
  rhs.communicate();
}

#endif // #ifndef DGRHS_HH
