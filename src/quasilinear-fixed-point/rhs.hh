#ifndef RHS_HH
#define RHS_HH

#include <dune/fem/quadrature/cachingquadrature.hh>

// assembleRHS
// -----------

template<class Model, class Function, class Neuman, class DiscreteFunction>
void assembleRHS(const Model &model, const Function &function,
    const Neuman &neuman, DiscreteFunction &rhs) {
  const bool neumanBnd = model.hasNeumanBoundary();
  rhs.clear();
  typedef typename DiscreteFunction::DiscreteFunctionSpaceType DiscreteFunctionSpaceType;

  typedef typename DiscreteFunctionSpaceType::GridPartType GridPartType;
  typedef typename DiscreteFunctionSpaceType::IteratorType IteratorType;
  typedef typename IteratorType::Entity EntityType;
  typedef typename EntityType::Geometry GeometryType;
  typedef typename GridPartType::IntersectionIteratorType IntersectionIteratorType;
  typedef typename IntersectionIteratorType::Intersection IntersectionType;

  const DiscreteFunctionSpaceType &dfSpace = rhs.space();

  typedef typename DiscreteFunctionSpaceType::GridPartType GridPartType;
  typedef Dune::Fem::CachingQuadrature<GridPartType, 0> QuadratureType;
  typedef Dune::Fem::ElementQuadrature<GridPartType, 1> FaceQuadratureType;
  const int quadOrder = 2 * dfSpace.order() + 1;

  Dune::Fem::TemporaryLocalFunction < DiscreteFunctionSpaceType
      > rhsLocal(dfSpace);

  for (const EntityType &entity : dfSpace) {
    const GeometryType geometry = entity.geometry();

    const typename Function::LocalFunctionType localFunction =
        function.localFunction(entity);

    rhsLocal.init(entity);
    rhsLocal.clear();

    typedef typename Function::RangeType RangeType;

    QuadratureType quadrature(entity, quadOrder);
    const size_t numQuadraturePoints = quadrature.nop();
    for (size_t pt = 0; pt < numQuadraturePoints; ++pt) {
      // obtain quadrature point
      const typename QuadratureType::CoordinateType &x = quadrature.point(
          pt);

      // evaluate f
      RangeType f;
      localFunction.evaluate(quadrature[pt], f);

      // multiply by quadrature weight
      f *= quadrature.weight(pt) * geometry.integrationElement(x);

      // add f * phi_i to rhsLocal[ i ]
      rhsLocal.axpy(quadrature[pt], f);
    }

    if (neumanBnd) {
      if (!entity.hasBoundaryIntersections())
        continue;

      const IntersectionIteratorType iitend = dfSpace.gridPart().iend(
          entity);
      for (IntersectionIteratorType iit = dfSpace.gridPart().ibegin(
          entity); iit != iitend; ++iit) // looping over intersections
          {
        const IntersectionType &intersection = *iit;
        if (!intersection.boundary())
          continue;
        Dune::FieldVector<bool, RangeType::dimension> components(true);
        // if ( model.isDirichletIntersection( intersection, components) )
        //   continue;
        bool hasDirichletComponent = model.isDirichletIntersection(
            intersection, components);

        const typename Neuman::LocalFunctionType neumanLocal =
            neuman.localFunction(entity);

        const typename IntersectionType::Geometry &intersectionGeometry =
            intersection.geometry();
        FaceQuadratureType quadInside(dfSpace.gridPart(), intersection,
            quadOrder, FaceQuadratureType::INSIDE);
        const size_t numQuadraturePoints = quadInside.nop();
        for (size_t pt = 0; pt < numQuadraturePoints; ++pt) {
          const typename FaceQuadratureType::LocalCoordinateType &x =
              quadInside.localPoint(pt);
          RangeType nval;
          neumanLocal.evaluate(quadInside[pt], nval);
          nval *= quadInside.weight(pt)
              * intersectionGeometry.integrationElement(x);
          for (int k = 0; k < RangeType::dimension; ++k)
            if (hasDirichletComponent && components[k])
              nval[k] = 0;
          rhsLocal.axpy(quadInside[pt], nval);
        }
      }
    }

    rhs.addLocalDofs(entity, rhsLocal.localDofVector());
  }
  rhs.communicate();
}

#endif // #ifndef RHS_HH
