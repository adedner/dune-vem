#ifndef DUNE_VEM_AGGLOMERATION_DGSPACE_HH
#define DUNE_VEM_AGGLOMERATION_DGSPACE_HH

#include <utility>

#include <dune/common/power.hh>
#include <dune/common/version.hh>

#if DUNE_VERSION_NEWER(DUNE_FEM, 2, 6)
#include <dune/fem/common/hybrid.hh>
#endif // #if DUNE_VERSION_NEWER(DUNE_FEM, 2, 6)

#include <dune/fem/space/common/commoperations.hh>
#include <dune/fem/space/common/defaultcommhandler.hh>
#include <dune/fem/space/common/discretefunctionspace.hh>
#include <dune/fem/space/common/functionspace.hh>
// #include <dune/fem/space/shapefunctionset/legendre.hh>
#include <dune/fem/space/shapefunctionset/orthonormal.hh>
#include <dune/fem/space/shapefunctionset/proxy.hh>
#include <dune/fem/space/shapefunctionset/vectorial.hh>
#include <dune/fem/solver/cginverseoperator.hh>
#include <dune/fem/operator/linear/spoperator.hh>

#include <dune/vem/agglomeration/basisfunctionset.hh>
#include <dune/vem/agglomeration/boundingbox.hh>
#include <dune/vem/agglomeration/dgmapper.hh>
#include <dune/vem/operator/mass.hh>
#include <dune/vem/misc/vector.hh>

namespace Dune
{

  namespace Vem
  {

    // Internal Forward Declarations
    // -----------------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    class AgglomerationDGSpace;

    // IsAgglomerationDGSpace
    // ------------------------------

    template< class DiscreteFunctionSpace >
    struct IsAgglomerationDGSpace
      : std::integral_constant< bool, false >
    {};

    template< class FunctionSpace, class GridPart, int order >
    struct IsAgglomerationDGSpace< AgglomerationDGSpace< FunctionSpace, GridPart, order > >
      : std::integral_constant< bool, true >
    {};


    // AgglomerationDGSpaceTraits
    // --------------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    struct AgglomerationDGSpaceTraits
    {
      friend class AgglomerationDGSpace< FunctionSpace, GridPart, polOrder >;

      typedef AgglomerationDGSpace< FunctionSpace, GridPart, polOrder > DiscreteFunctionSpaceType;

      typedef FunctionSpace FunctionSpaceType;
      typedef GridPart GridPartType;

      static const int codimension = 0;

    private:
      typedef typename GridPartType::template Codim< codimension >::EntityType EntityType;

      /*
      typedef typename Fem::FunctionSpace< typename FunctionSpaceType::DomainFieldType, typename FunctionSpaceType::RangeFieldType, FunctionSpaceType::dimDomain, 1 > ScalarFunctionSpaceType;
      // typedef Fem::LegendreShapeFunctionSet< ScalarFunctionSpaceType > ScalarShapeFunctionSetType;
      typedef Fem::OrthonormalShapeFunctionSet< ScalarFunctionSpaceType > ScalarShapeFunctionSetType;
      */

    public:
      typedef Dune::Fem::FunctionSpace<
          typename FunctionSpace::DomainFieldType, typename FunctionSpace::RangeFieldType,
           GridPartType::dimension, 1
        > ScalarShapeFunctionSpaceType;

      struct ScalarShapeFunctionSet
        : public Dune::Fem::OrthonormalShapeFunctionSet< ScalarShapeFunctionSpaceType >
        // : public Dune::Fem::LegendreShapeFunctionSet< ScalarShapeFunctionSpaceType,true >
      {
        typedef Dune::Fem::OrthonormalShapeFunctionSet< ScalarShapeFunctionSpaceType >   BaseType;
        // typedef Dune::Fem::LegendreShapeFunctionSet< ScalarShapeFunctionSpaceType,true >   BaseType;

        static constexpr int numberShapeFunctions =
              Dune::Fem::OrthonormalShapeFunctions< ScalarShapeFunctionSpaceType::dimDomain >::size(polOrder);
      public:
        explicit ScalarShapeFunctionSet ( Dune::GeometryType type )
          : BaseType( type, polOrder )
        {
          assert( size() == BaseType::size() );
        }
        explicit ScalarShapeFunctionSet ( Dune::GeometryType type, int p )
          : BaseType( type, p )
        {
          assert( size() == BaseType::size() );
        }

        // overload size method because it's a static value
        static constexpr unsigned int size() { return numberShapeFunctions; }
      };
      typedef ScalarShapeFunctionSet ScalarShapeFunctionSetType;
      typedef Fem::VectorialShapeFunctionSet< Fem::ShapeFunctionSetProxy< ScalarShapeFunctionSetType >, typename FunctionSpaceType::RangeType > ShapeFunctionSetType;

      typedef BoundingBoxBasisFunctionSet< GridPartType, ShapeFunctionSetType > BasisFunctionSetType;

      // static const std::size_t localBlockSize = FunctionSpaceType::dimRange * StaticPower< polOrder+1, GridPartType::dimension >::power;
      typedef Hybrid::IndexRange< int, FunctionSpaceType::dimRange * ScalarShapeFunctionSet::numberShapeFunctions > LocalBlockIndices;
      typedef AgglomerationDGMapper< GridPartType > BlockMapperType;

      template< class DiscreteFunction, class Operation = Fem::DFCommunicationOperation::Copy >
      struct CommDataHandle
      {
        typedef Operation OperationType;
        typedef Fem::DefaultCommunicationHandler< DiscreteFunction, Operation > Type;
      };
    };



    // AgglomerationDGSpace
    // --------------------

    template< class FunctionSpace, class GridPart, int polOrder >
    class AgglomerationDGSpace
      : public Fem::DiscreteFunctionSpaceDefault< AgglomerationDGSpaceTraits< FunctionSpace, GridPart, polOrder > >
    {
      typedef AgglomerationDGSpace< FunctionSpace, GridPart, polOrder > ThisType;
      typedef Fem::DiscreteFunctionSpaceDefault< AgglomerationDGSpaceTraits< FunctionSpace, GridPart, polOrder > > BaseType;

    public:
      typedef typename BaseType::Traits Traits;

      typedef Agglomeration< GridPart > AgglomerationType;

      typedef typename BaseType::BasisFunctionSetType BasisFunctionSetType;

      typedef typename BaseType::BlockMapperType BlockMapperType;

      typedef typename BaseType::EntityType EntityType;
      typedef typename BaseType::GridPartType GridPartType;

      enum { hasLocalInterpolate = false };

      AgglomerationDGSpace ( AgglomerationType &agglomeration )
        : BaseType( agglomeration.gridPart() ),
          blockMapper_( agglomeration ),
          scalarShapeFunctionSet_( Dune::GeometryType(Dune::GeometryType::cube,GridPart::dimension), polOrder ),
          counter_(0)
      {
        blockMapper().agglomeration().onbBasis(polOrder);
      }
      void update()
      {
        ++counter_;
        if (agglomeration().counter()<counter_)
          blockMapper().agglomeration().update();
      }

      const AgglomerationType &agglomeration() const { return blockMapper().agglomeration(); }
      AgglomerationType &agglomeration() { return blockMapper().agglomeration(); }
      const BoundingBox< GridPart >& boundingBox( const EntityType &entity ) const
      {
        return agglomeration().boundingBox(agglomeration().index(entity));
      }

      const BasisFunctionSetType basisFunctionSet ( const EntityType &entity ) const
      {
        const std::size_t agglomerate = agglomeration().index(entity);
        typename Traits::ShapeFunctionSetType scalarShapeFunctionSet( &scalarShapeFunctionSet_ );
        return BasisFunctionSetType(entity, agglomerate,
                                    agglomeration().boundingBoxes(), true, std::move(scalarShapeFunctionSet));
      }

      BlockMapperType &blockMapper () const { return blockMapper_; }

      // extra interface methods

      static constexpr bool continuous () noexcept { return false; }

      static constexpr bool continuous ( const typename BaseType::IntersectionType & ) noexcept { return false; }

      static constexpr int order ( const EntityType & ) noexcept { return polOrder; }
      static constexpr int order () { return polOrder; }

      static constexpr Fem::DFSpaceIdentifier type () noexcept { return Fem::GenericSpace_id; }

    private:
      mutable BlockMapperType blockMapper_;
      Std::vector< BoundingBox< GridPart > > boundingBoxes_;
      typename Traits::ScalarShapeFunctionSetType scalarShapeFunctionSet_;
      std::size_t counter_;
    };

    template <class DFSpace>
    struct BBDGPenalty
    {
      BBDGPenalty(const DFSpace &space, double penalty)
      : space_(space)
      , penalty_(penalty)
      {}
      // implement: h_e = min(|e^+|,|e^-|) / |e|
      template <class Intersection>
      double operator()(const Intersection &intersection,
                        double intersectionArea, double area, double nbArea) const
      {
        const auto &bbIn = space_.boundingBox(intersection.inside());
        auto delta = bbIn.second - bbIn.first;
        auto volume = bbIn.volume();
        if (intersection.neighbor())
        {
          const auto &bbOut  = space_.boundingBox(intersection.outside());
          delta[0] = std::min(delta[0], bbOut.second[0]-bbOut.first[0]);
          delta[1] = std::min(delta[1], bbOut.second[1]-bbOut.first[1]);
          volume = std::min(volume,bbOut.volume());
        }
        /*
        auto normal = intersection.unitOuterNormal({0.5});
        normal[0] = std::abs(normal[0]);
        normal[1] = std::abs(normal[1]);
        double h = 0;
        if (normal[0]<1e-10)
          h = delta[1];
        else if (normal[1]<1e-10)
          h = delta[0];
        else
          h = std::min(delta[0]/normal[0], delta[1]/normal[1]);
        std::cout << "n=" << normal[0] << "," << normal[1];
        std::cout << "    bbox=" << bbIn.first << "," << bbIn.second;
        std::cout << "    delta=" << delta[0] << " " << delta[1];
        std::cout << "    h=" << h << std::endl;
        return penalty_ / h;
        */
        // const double hInv = intersectionArea / std::min( area, nbArea );
        const double hInv = intersectionArea / volume;
        return penalty_ * hInv;
      }
      const double &factor() const { return penalty_; }
      private:
      const DFSpace &space_;
      double penalty_;
    };

  } // namespace Vem

  namespace Fem
  {
    template< class GridFunction, class DiscreteFunction, unsigned int partitions >
    static inline std::enable_if_t< std::is_convertible< GridFunction, HasLocalFunction >::value && Dune::Vem::IsAgglomerationDGSpace< typename DiscreteFunction::DiscreteFunctionSpaceType >::value >
    interpolate ( const GridFunction &u, DiscreteFunction &v, PartitionSet< partitions > ps )
    {
      // !!! a very crude implementation - should be done locally on each polygon
      v.clear();
      typedef AdaptiveDiscreteFunction<typename DiscreteFunction::DiscreteFunctionSpaceType> DF;
      DF rhs( "rhs", v.space() );
      DF vtmp( "sol", v.space() );
      rhs.clear();
      Dune::Vem::applyMass( u, rhs );
      typedef Dune::Fem::SparseRowLinearOperator< DF, DF > LinearOperator;
      LinearOperator assembledMassOp( "assembled mass operator", v.space(), v.space() );
      Dune::Vem::MassOperator< LinearOperator > massOp( v.space() );
      massOp.jacobian( vtmp, assembledMassOp );
      auto param = Dune::Fem::parameterDict("fem.solver.", "verbose",false,
          "preconditioning.method","jacobi",
          "errorMeasure","absolute",
          "tolerance",1e-10);
      Dune::Fem::CGInverseOperator< DF > invOp(param);
      invOp.bind(assembledMassOp);
      vtmp.clear();
      invOp( rhs, vtmp );
      v.assign(vtmp);
    }

  } // namespace Fem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_DGSPACE_HH
