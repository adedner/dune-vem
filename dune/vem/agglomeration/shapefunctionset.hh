#ifndef DUNE_VEM_AGGLOMERATION_SHAPEFUNCTIONSET_HH
#define DUNE_VEM_AGGLOMERATION_SHAPEFUNCTIONSET_HH

#include <cassert>
#include <cstddef>

#include <utility>

#include <dune/fem/quadrature/quadrature.hh>
#include <dune/fem/space/shapefunctionset/vectorial.hh>

namespace Dune
{

  namespace Vem
  {

    // BoundingBoxShapeFunctionSet
    // ---------------------------

    template< class Entity, class ShapeFunctionSet >
    class BoundingBoxBasisFunctionSet
    {
      typedef BoundingBoxShapeFunctionSet< Entity, ShapeFunctionSet > ThisType;

    public:
      typedef typename ShapeFunctionSet::FunctionSpaceType FunctionSpaceType;

      typedef typename FunctionSpaceType::DomainFieldType DomainFieldType;
      typedef typename FunctionSpaceType::RangeFieldType RangeFieldType;

      typedef typename FunctionSpaceType::DomainType DomainType;
      typedef typename FunctionSpaceType::RangeType RangeType;
      typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;
      typedef typename FunctionSpaceType::HessianRangeType HessianRangeType;

      static constexpr int dimDomain = DomainType::dimension;

    private:
      template< class Functor >
      struct JacobianFunctor
      {
        explicit JacobianFunctor ( const std::pair< DomainType, DomainType > &bbox, Functor functor )
          : extent( bbox.second - bbox.first ), functor_( functor )
        {}

        void operator() ( std::size_t i, JacobianRangeType jacobian )
        {
          for( int i = 0; i < RangeType::dimension; ++i )
            applyScalar( jacobian[ i ] );
          functor_( i, jacobian );
        }

        template< class ScalarJacobian >
        void operator() ( std::size_t i, Fem::MakeVectorialExpression< ScalarJacobian, JacobianRangeType > jacobian )
        {
          applyScalar( jacobian.scalar()[ 0 ] );
          functor_( i, jacobian );
        }

      private:
        void applyScalar ( FieldVector< RangeFieldType, dimDomain > &jacobian ) const
        {
          for( int j = 0; j < dimDomain; ++j )
            jacobian[ j ] /= extent[ j ]_;
        }

        DomainType extent_;
        Functor functor_;
      };

      template< class Functor >
      struct HessianFunctor
      {
        explicit HessianFunctor ( const std::pair< DomainType, DomainType > &bbox, Functor functor )
          : extent( bbox.second - bbox.first ), functor_( functor )
        {}

        void operator() ( std::size_t i, HessianRangeType hessian )
        {
          for( int i = 0; i < RangeType::dimension; ++i )
            applyScalar( hessian[ i ] );
          functor_( i, hessian );
        }

        template< class ScalarHessian >
        void operator() ( std::size_t i, Fem::MakeVectorialExpression< ScalarHessian, HessianRangeType > hessian )
        {
          applyScalar( hessian.scalar()[ 0 ] );
          functor_( i, hessian );
        }

      private:
        void applyScalar ( FieldMatrix< RangeFieldType, dimDomain, dimDomain > &hessian ) const
        {
          for( int j = 0; j < dimDomain; ++j )
            for( int k = 0; k < dimDomain; ++k )
              hessian[ j ][ k ] /= (extent_[ j ] * extent_[ k ]);
        }

        DomainType extent_;
      };

    public:
      BoundingBoxShapeFunctionSet () = default;

      BoundingBoxShapeFunctionSet ( const EntityType &entity, std::pair< DomainType, DomainType > bbox,
                                    ShapeFunctionSet shapeFunctionSet = ShapeFunctionSet() )
        : entity_( &entity ), shapeFunctionSet_( std::move( shapeFunctionSet ) ), bbox_( std::move( bbox ) )
      {}

      int order () const { return shapeFunctionSet_.order(); }

      std::size_t size () const { return shapeFunctionSet_.size(); }

      template< class Point, class Functor >
      void evaluateEach ( const Point &x, Functor functor ) const
      {
        shapeFunctionSet_.evaluateEach( position( x ), functor );
      }

      template< class Point, class Functor >
      void jacobianEach ( const Point &x, Functor functor ) const
      {
        shapeFunctionSet_.jacobianEach( position( x ), JacobianFunctor< Functor >( bbox_, functor ) );
      }

      template< class Point, class Functor >
      void hessianEach ( const Point &x, Functor functor ) const
      {
        shapeFunctionSet_.hessianEach( position( x ), HessianFunctor< Functor >( bbox_, functor ) );
      }

      const EntityType &entity () const { assert( entity_ ); return *entity_; }

    private:
      template< class Point >
      DomainType position ( const Point &x ) const
      {
        DomainType y = entity().geometry().global( Fem::coordinate( x ) ) - bbox_.first;
        for( int k = 0; k < dimDomain; ++k )
          y[ k ] /= (bbox_.second[ k ] - bbox_.first[ k ]);
        return y;
      }

      const EntityType *entity_ = nullptr;
      ShapeFunctionSet shapeFunctionSet_;
      std::pair< DomainType, DomainType > bbox_;
    };

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_SHAPEFUNCTIONSET_HH
