#ifndef DUNE_VEM_SPACE_BASISFUNCTIONSET_HH
#define DUNE_VEM_SPACE_BASISFUNCTIONSET_HH

#include <cassert>
#include <cstddef>

#include <algorithm>
#include <type_traits>
#include <utility>

#include <dune/geometry/referenceelements.hh>

#include <dune/fem/common/fmatrixcol.hh>
#include <dune/fem/quadrature/quadrature.hh>
#include <dune/fem/space/basisfunctionset/functor.hh>
#include <dune/fem/space/shapefunctionset/vectorial.hh>

#include <dune/vem/agglomeration/functor.hh>

namespace Dune
{

  namespace Vem
  {

    // VEMBasisFunctionSet
    // -------------------

    // TODO: add template arguments for ValueProjection and JacobianProjection
    template< class Entity, class ShapeFunctionSet >
    class VEMBasisFunctionSet
    {
      typedef VEMBasisFunctionSet< Entity, ShapeFunctionSet > ThisType;

    public:
      typedef Entity EntityType;

      typedef typename ShapeFunctionSet::FunctionSpaceType FunctionSpaceType;

      typedef typename FunctionSpaceType::DomainFieldType DomainFieldType;
      typedef typename FunctionSpaceType::RangeFieldType RangeFieldType;

      typedef typename FunctionSpaceType::DomainType DomainType;
      typedef typename FunctionSpaceType::RangeType RangeType;
      typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;
      typedef typename FunctionSpaceType::HessianRangeType HessianRangeType;

      static constexpr int dimDomain = DomainType::dimension;
      static constexpr int dimRange  = RangeType::dimension;

      typedef ReferenceElement< typename DomainType::field_type, dimDomain > ReferenceElementType;

      typedef std::vector< std::vector< DomainFieldType > > ValueProjection;
      typedef std::vector< std::vector< DomainType > > JacobianProjection;
      typedef std::vector< std::vector< FieldMatrix < DomainFieldType, dimDomain, dimDomain > > > HessianProjection;


      VEMBasisFunctionSet () = default;

      VEMBasisFunctionSet ( const EntityType &entity, std::pair< DomainType, DomainType > bbox,
                            ValueProjection valueProjection, JacobianProjection jacobianProjection,
                            HessianProjection hessianProjection,
                            ShapeFunctionSet shapeFunctionSet = ShapeFunctionSet() )
        : entity_( &entity ), //polygon
          shapeFunctionSet_( std::move( shapeFunctionSet ) ),
          valueProjection_( std::move( valueProjection ) ),
          jacobianProjection_( std::move( jacobianProjection ) ),
          hessianProjection_( std::move( hessianProjection ) ),
          bbox_( std::move( bbox ) ),
          size_( valueProjection_[0].size() * dimRange)
      {}

      int order () const { return shapeFunctionSet_.order(); }

      std::size_t size () const { return size_; }

      const ReferenceElementType &referenceElement () const
      {
        return ReferenceElements< DomainFieldType, dimDomain >::general( entity().type() );
      }

      template< class Quadrature, class Vector, class DofVector >
      void axpy ( const Quadrature &quadrature, const Vector &values, DofVector &dofs ) const
      {
        const std::size_t nop = quadrature.nop();
        for( std::size_t qp = 0; qp < nop; ++qp )
          axpy( quadrature[ qp ], values[ qp ], dofs );
      }

      template< class Quadrature, class VectorA, class VectorB, class DofVector >
      void axpy ( const Quadrature &quadrature, const VectorA &valuesA, const VectorB &valuesB, DofVector &dofs ) const
      {
        const std::size_t nop = quadrature.nop();
        for( std::size_t qp = 0; qp < nop; ++qp )
        {
          axpy( quadrature[ qp ], valuesA[ qp ], dofs );
          axpy( quadrature[ qp ], valuesB[ qp ], dofs );
        }
      }

      template< class Point, class DofVector >
      void axpy ( const Point &x, const RangeType &valueFactor, DofVector &dofs ) const
      {
        shapeFunctionSet_.evaluateEach( position( x ), [ this, &valueFactor, &dofs ] ( std::size_t alpha, RangeType phi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              dofs[ j ] += valueProjection_[ alpha ][ j ] * (valueFactor * phi_alpha);
          } );
      }

      template< class Point, class DofVector >
      void axpy ( const Point &x, const JacobianRangeType &jacobianFactor, DofVector &dofs ) const
      {
        shapeFunctionSet_.evaluateEach( position( x ), [ this, &jacobianFactor, &dofs ] ( std::size_t alpha, RangeType phi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              for( int k = 0; k < dimDomain; ++k )
              {
                FieldMatrixColumn< const JacobianRangeType > jacobianFactor_k( jacobianFactor, k );
                dofs[ j ] += jacobianProjection_[ alpha ][ j ][ k ] * (jacobianFactor_k * phi_alpha);
              }
          } );
      }

      template< class Point, class DofVector >
      void axpy ( const Point &x, const RangeType &valueFactor, const JacobianRangeType &jacobianFactor, DofVector &dofs ) const
      {
        axpy( x, valueFactor, dofs );
        axpy( x, jacobianFactor, dofs );
      }

      template< class Quadrature, class DofVector, class Values >
      void evaluateAll ( const Quadrature &quadrature, const DofVector &dofs, Values &values ) const
      {
        const std::size_t nop = quadrature.nop();
        for( std::size_t qp = 0; qp < nop; ++qp )
          evaluateAll( quadrature[ qp ], dofs, values[ qp ] );
      }

      template< class Point, class DofVector >
      void evaluateAll ( const Point &x, const DofVector &dofs, RangeType &value ) const
      {
        value = RangeType( 0 );
        shapeFunctionSet_.evaluateEach( position( x ), [ this, &dofs, &value ] ( std::size_t alpha, RangeType phi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
            {
              double v = valueProjection_[ alpha ][ j ];
              value.axpy( valueProjection_[ alpha ][ j ]*dofs[ j ], phi_alpha );
            }
          } );
      }

      template< class Point, class Values > const
      void evaluateAll ( const Point &x, Values &values ) const
      {
        assert( values.size() >= size() );
        std::fill( values.begin(), values.end(), RangeType( 0 ) );
        shapeFunctionSet_.evaluateEach( position(x), [ this, &values ] ( std::size_t alpha, RangeType phi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              values[ j ].axpy( valueProjection_[ alpha ][ j ], phi_alpha );
          } );
      }

      // TODO: use lower order shape function set for Jacobian
      template< class Quadrature, class DofVector, class Jacobians >
      void jacobianAll ( const Quadrature &quadrature, const DofVector &dofs, Jacobians &jacobians ) const
      {
        const std::size_t nop = quadrature.nop();
        for( std::size_t qp = 0; qp < nop; ++qp )
          jacobianAll( quadrature[ qp ], dofs, jacobians[ qp ] );
      }

      template< class Point, class DofVector >
      void jacobianAll ( const Point &x, const DofVector &dofs, JacobianRangeType &jacobian ) const
      {
        jacobian = JacobianRangeType( 0 );
        shapeFunctionSet_.evaluateEach( position( x ), [ this, &dofs, &jacobian ] ( std::size_t alpha, RangeType phi_alpha ) {
            const auto &jacobianProjectionAlpha = jacobianProjection_[alpha];
            for( std::size_t j = 0; j < size(); ++j )
              jacobian[0].axpy( dofs[j]*phi_alpha[0], jacobianProjectionAlpha[j]);
          } );
      }

      template< class Point, class Jacobians > const
      void jacobianAll ( const Point &x, Jacobians &jacobians ) const
      {
        assert( jacobians.size() >= size() );
        std::fill( jacobians.begin(), jacobians.end(), JacobianRangeType( 0 ) );
        shapeFunctionSet_.evaluateEach( position(x), [ this, &jacobians ] ( std::size_t alpha, RangeType phi_alpha ) {
            const auto &jacobianProjectionAlpha = jacobianProjection_[alpha];
            for( std::size_t j = 0; j < size(); ++j )
              jacobians[j][0].axpy( phi_alpha[0], jacobianProjectionAlpha[j]);
        } );
      }

      template< class Quadrature, class DofVector, class Hessians >
      void hessianAll ( const Quadrature &quadrature, const DofVector &dofs, Hessians &hessians ) const
      {
        const std::size_t nop = quadrature.nop();
        for(std::size_t qp = 0; qp <nop; ++qp )
          hessianAll( quadrature[ qp ], dofs, hessians[ qp ] );
      }

      template< class Point, class DofVector >
      void hessianAll ( const Point &x, const DofVector &dofs, HessianRangeType &hessian ) const
      {
        hessian = HessianRangeType( 0 );
        shapeFunctionSet_.evaluateEach( position(x), [this, &dofs, &hessian ] ( std::size_t alpha, RangeType phi_alpha ) {
            const auto &hessianProjectionAlpha = hessianProjection_[alpha];
            for( std::size_t j = 0; j < size(); ++j )
              hessian[0].axpy( dofs[j]*phi_alpha[0], hessianProjectionAlpha[j]);
        } );
      }

      template< class Point, class Hessians > const
      void hessianAll ( const Point &x, Hessians &hessians ) const
      {
        assert( hessians.size() >= size() );
        std::fill( hessians.begin(), hessians.end(), HessianRangeType( 0 ) );
        shapeFunctionSet_.evaluateEach( position(x), [ this, &hessians ] ( std::size_t alpha, RangeType phi_alpha ) {
            const auto &hessianProjectionAlpha = hessianProjection_[alpha];
            for( std::size_t j = 0; j < size(); ++j )
              hessians[j][0].axpy( phi_alpha[0], hessianProjectionAlpha[j]);
        } );
      }

      const EntityType &entity () const { assert( entity_ ); return *entity_; }

      template< class Point, class Factor >
      void axpy ( const Point &x, const Factor &factor, DynamicVector<DomainType> &dofs ) const
      {
        shapeFunctionSet_.evaluateEach( position( x ), [ this, &factor, &dofs ] ( std::size_t alpha, RangeType phi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              dofs[ j ].axpy( phi_alpha[0]*valueProjection_[ alpha ][ j ], factor );
          } );
      }

    private:
      template< class Point >
      DomainType position ( const Point &x ) const
      {
        return Fem::coordinate(x);
      }

      const EntityType *entity_ = nullptr;
      ShapeFunctionSet shapeFunctionSet_;
      ValueProjection valueProjection_;
      JacobianProjection jacobianProjection_;
      HessianProjection hessianProjection_;
      std::pair< DomainType, DomainType > bbox_;
      size_t size_;
    };

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_BASISFUNCTIONSET_HH
