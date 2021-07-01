#ifndef DUNE_VEM_OPERATOR_MASS_HH
#define DUNE_VEM_OPERATOR_MASS_HH

#include <utility>
#include <vector>

#include <dune/grid/common/rangegenerators.hh>

#include <dune/fem/function/localfunction/temporary.hh>
#include <dune/fem/operator/common/stencil.hh>
#include <dune/fem/operator/common/differentiableoperator.hh>
#include <dune/fem/quadrature/cachingquadrature.hh>

namespace Dune
{

  namespace Vem
  {

    // applyMass
    // ---------

    template< class GridFunction, class DiscreteFunction >
    inline static void applyMass ( const GridFunction &u, DiscreteFunction &w )
    {
      w.clear();

      Fem::TemporaryLocalFunction< typename DiscreteFunction::DiscreteFunctionSpaceType > wLocal( w.space() );
      Fem::ConstLocalFunction<GridFunction> uLocal(u);
      for( const auto &entity : elements( w.gridPart() ) )
      {
        const auto geometry = entity.geometry();

        uLocal.bind(entity);

        wLocal.init( entity );
        wLocal.clear();

        const Fem::CachingQuadrature< typename GridFunction::GridPartType, 0 > quadrature( entity, uLocal.order() + wLocal.order() );
        const std::size_t numQuadraturePoints = quadrature.nop();
        for( std::size_t qp = 0; qp < numQuadraturePoints; ++qp )
        {
          typename GridFunction::RangeType uValue;
          uLocal.evaluate( quadrature[ qp ], uValue );
          uValue *= quadrature.weight( qp ) * geometry.integrationElement( quadrature.point( qp ) );
          wLocal.axpy( quadrature[ qp ], uValue );
        }

        w.addLocalDofs( entity, wLocal.localDofVector() );
        uLocal.unbind();
      }

      w.communicate();
    }



    // MassOperator
    // ------------

    template< class JacobianOperator >
    class MassOperator
      : public Fem::DifferentiableOperator< JacobianOperator >
    {
      typedef MassOperator< JacobianOperator > ThisType;
      typedef Fem::DifferentiableOperator< JacobianOperator > BaseType;

    public:
      typedef typename BaseType::DomainFunctionType DomainFunctionType;
      typedef typename BaseType::RangeFunctionType RangeFunctionType;
      typedef typename BaseType::JacobianOperatorType JacobianOperatorType;

      typedef typename RangeFunctionType::DiscreteFunctionSpaceType DiscreteFunctionSpaceType;

      explicit MassOperator ( const DiscreteFunctionSpaceType &dfSpace )
        : dfSpace_( dfSpace ), stencil_( dfSpace, dfSpace )
      {}

      void operator() ( const DomainFunctionType &u, RangeFunctionType &w ) const { applyMass( u, w ); }

      void jacobian ( const DomainFunctionType &u, JacobianOperatorType &jOp ) const;

    private:
      const DiscreteFunctionSpaceType &dfSpace_;
      Fem::DiagonalStencil< DiscreteFunctionSpaceType, DiscreteFunctionSpaceType > stencil_;
    };



    // Implementation of MassOperator
    // ------------------------------

    template< class JacobianOperator >
    inline void MassOperator< JacobianOperator >
      ::jacobian ( const DomainFunctionType &u, JacobianOperatorType &jOp ) const
    {
      jOp.reserve( stencil_ );
      jOp.clear();
      std::ofstream fMassGlobal("./mass-agglo.dat");
      const std::size_t maxNumLocalDofs = DiscreteFunctionSpaceType::localBlockSize * u.space().blockMapper().maxNumDofs();
      std::vector< typename DiscreteFunctionSpaceType::RangeType > values( maxNumLocalDofs );

      for( const auto &entity : elements( u.gridPart() ) )
      {
        const auto geometry = entity.geometry();

        auto localMatrix = jOp.localMatrix( entity, entity );

        const auto &basis = localMatrix.domainBasisFunctionSet();
        const std::size_t numBasisFunctions = basis.size();

        const Fem::CachingQuadrature< typename DiscreteFunctionSpaceType::GridPartType, 0 > quadrature( entity, 2*basis.order() );
        const std::size_t numQuadraturePoints = quadrature.nop();
        for( std::size_t qp = 0; qp < numQuadraturePoints; ++qp )
        {
          basis.evaluateAll( quadrature[ qp ], values );
          const typename DiscreteFunctionSpaceType::DomainFieldType weight
            = quadrature.weight( qp ) * geometry.integrationElement( quadrature.point( qp ) );
          for( unsigned int i = 0; i < numBasisFunctions; ++i )
            localMatrix.column( i ).axpy( values, values[ i ], weight );
        }
      }
      // jOp.exportMatrix().print(fMassGlobal);
      jOp.flushAssembly();
    }

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_OPERATOR_MASS_HH
