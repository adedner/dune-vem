#ifndef DUNE_VEM_SPACE_HK_HH
#define DUNE_VEM_SPACE_HK_HH

#include <cassert>
#include <utility>

#include <dune/fem/space/common/defaultcommhandler.hh>
#include <dune/fem/space/common/discretefunctionspace.hh>
#include <dune/fem/space/common/functionspace.hh>
#include <dune/fem/space/common/capabilities.hh>
#include <dune/vem/misc/compatibility.hh>

#include <dune/vem/space/interpolate.hh>
#include <dune/vem/space/default.hh>

#include <dune/vem/space/hkinterpolation.hh>

namespace Dune
{
  namespace Vem
  {
    // Internal Forward Declarations
    // -----------------------------

    template<class FunctionSpace, class GridPart, bool vectorSpace = false>
    class AgglomerationVEMSpace;

    // IsAgglomerationVEMSpace
    // -----------------------

    template<class DiscreteFunctionSpace>
    struct IsAgglomerationVEMSpace
            : std::integral_constant<bool, false> {
    };

    template<class FunctionSpace, class GridPart, bool vectorSpace>
    struct IsAgglomerationVEMSpace<AgglomerationVEMSpace<FunctionSpace, GridPart,vectorSpace> >
            : std::integral_constant<bool, true> {
    };

    // AgglomerationVEMSpaceTraits
    // ---------------------------

    template<class FunctionSpace, class GridPart, bool vectorspace>
    struct AgglomerationVEMSpaceTraits
    {
      static const bool vectorSpace = vectorspace;
      friend class AgglomerationVEMSpace<FunctionSpace, GridPart, vectorSpace>;

      typedef AgglomerationVEMSpace<FunctionSpace, GridPart, vectorSpace> DiscreteFunctionSpaceType;

      typedef GridPart GridPartType;

      static const int dimension = GridPartType::dimension;
      static const int codimension = 0;
      static const int dimDomain = FunctionSpace::DomainType::dimension;
      static const int dimRange = FunctionSpace::RangeType::dimension;
      static const int baseRangeDimension = vectorSpace ? dimRange : 1;

      typedef typename GridPartType::template Codim<codimension>::EntityType EntityType;
      typedef FunctionSpace FunctionSpaceType;

      // a scalar function space
      typedef Dune::Fem::FunctionSpace<
              typename FunctionSpace::DomainFieldType, typename FunctionSpace::RangeFieldType,
              GridPartType::dimension, 1 > ScalarFunctionSpaceType;

      // scalar BB basis
      typedef Dune::Fem::OrthonormalShapeFunctionSet< ScalarFunctionSpaceType > ScalarShapeFunctionSetType;
      typedef BoundingBoxBasisFunctionSet< GridPartType, ScalarShapeFunctionSetType > ScalarBBBasisFunctionSetType;

      // vector version of the BB basis for use with vector spaces
      typedef std::conditional_t< vectorSpace,
              Fem::VectorialShapeFunctionSet< ScalarBBBasisFunctionSetType,typename FunctionSpace::RangeType>,
              // Fem::VectorialShapeFunctionSet< ScalarBBBasisFunctionSetType,typename ScalarFunctionSpaceType::RangeType>
              ScalarBBBasisFunctionSetType
              > BBBasisFunctionSetType;

      // vem basis function sets
      typedef VEMBasisFunctionSet <EntityType, BBBasisFunctionSetType> ScalarBasisFunctionSetType;
      typedef std::conditional_t< vectorSpace,
              ScalarBasisFunctionSetType,
              Fem::VectorialBasisFunctionSet<ScalarBasisFunctionSetType, typename FunctionSpaceType::RangeType>
              > BasisFunctionSetType;

      // Next we define test function space for the edges
      typedef Dune::Fem::FunctionSpace<double,double,GridPartType::dimensionworld-1,1> EdgeFSType;
      typedef Dune::Fem::OrthonormalShapeFunctionSet<EdgeFSType> ScalarEdgeShapeFunctionSetType;
      typedef Fem::VectorialShapeFunctionSet< ScalarEdgeShapeFunctionSetType,
              typename BBBasisFunctionSetType::RangeType> EdgeShapeFunctionSetType;

      // types for the mapper
      typedef Hybrid::IndexRange<int, FunctionSpaceType::dimRange> LocalBlockIndices;
      typedef VemAgglomerationIndexSet <GridPartType> IndexSetType;
      typedef AgglomerationDofMapper <GridPartType, IndexSetType> BlockMapperType;

      template<class DiscreteFunction, class Operation = Fem::DFCommunicationOperation::Copy>
      struct CommDataHandle {
          typedef Operation OperationType;
          typedef Fem::DefaultCommunicationHandler <DiscreteFunction, Operation> Type;
      };

      template <class T>
      using InterpolationType = AgglomerationVEMInterpolation<T>;
    };

    // AgglomerationVEMSpace
    // ---------------------
    template<class FunctionSpace, class GridPart, bool vectorSpace>
    struct AgglomerationVEMSpace
    : public DefaultAgglomerationVEMSpace< AgglomerationVEMSpaceTraits<FunctionSpace,GridPart,vectorSpace> >
    {
      typedef DefaultAgglomerationVEMSpace< AgglomerationVEMSpaceTraits<FunctionSpace,GridPart,vectorSpace> > BaseType;
      using BaseType::BaseType;
    };

  } // namespace Vem

  namespace Fem
  {
    namespace Capabilities
    {
        template<class FunctionSpace, class GridPart, bool vectorSpace>
        struct hasInterpolation<Vem::AgglomerationVEMSpace<FunctionSpace, GridPart, vectorSpace> > {
            static const bool v = false;
        };
    }
  } // namespace Fem
} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_HK_HH
