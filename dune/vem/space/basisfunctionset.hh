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
#include <dune/fem/storage/entitygeometry.hh>

#include <dune/vem/agglomeration/functor.hh>
#include <dune/vem/misc/vector.hh>
#include <dune/common/gmpfield.hh>

namespace Dune
{

  namespace Vem
  {

    // VEMBasisFunctionSet
    // -------------------

    // TODO: add template arguments for ValueProjection and JacobianProjection
    template< class Entity, class ShapeFunctionSet,
              class InterpolationType, class StorageField >
    class VEMBasisFunctionSet
      : public Dune::Fem::EntityGeometryStorage< Entity >
    {
      typedef Dune::Fem::EntityGeometryStorage< Entity > BaseType;
      typedef VEMBasisFunctionSet< Entity, ShapeFunctionSet, InterpolationType, StorageField > ThisType;

    public:
      typedef typename BaseType::EntityType  EntityType;
      typedef typename BaseType::Geometry    Geometry;

      typedef typename ShapeFunctionSet::FunctionSpaceType FunctionSpaceType;

      typedef StorageField StorageFieldType;
      typedef double DomainFieldType;
      typedef typename FunctionSpaceType::RangeFieldType RangeFieldType;
      typedef typename FunctionSpaceType::DomainType DomainType;
      typedef typename FunctionSpaceType::RangeType RangeType;
      typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;
      typedef typename FunctionSpaceType::HessianRangeType HessianRangeType;

      static constexpr int dimDomain = DomainType::dimension;
      static constexpr int dimRange  = RangeType::dimension;

      // typedef ReferenceElements< typename DomainType::field_type, dimDomain > ReferenceElementType;
      //typedef typename ReferenceElements< typename DomainType::field_type, dimDomain >::ReferenceElement ReferenceElementType;

      const auto& valueProjection() const { return (*valueProjections_)[agglomerate_]; }
      const auto& jacobianProjection() const { return (*jacobianProjections_)[agglomerate_]; }
      const auto& hessianProjection() const { return (*hessianProjections_)[agglomerate_]; }
      typedef Std::vector< Std::vector< StorageFieldType > > ValueProjection;
      typedef Std::vector< Std::vector< StorageFieldType > > JacobianProjection;
      typedef Std::vector< Std::vector< StorageFieldType > > HessianProjection;
      template <class T>
      using Vector = Std::vector<T>;

      VEMBasisFunctionSet () = default;

      VEMBasisFunctionSet ( const EntityType &entity,
                            int agglomerate,
                            std::shared_ptr<Vector<ValueProjection>> valueProjections,
                            std::shared_ptr<Vector<JacobianProjection>> jacobianProjections,
                            std::shared_ptr<Vector<HessianProjection>> hessianProjections,
                            ShapeFunctionSet shapeFunctionSet,
                            std::shared_ptr<InterpolationType> interpolation
                          )
        : BaseType( entity ),
          agglomerate_(agglomerate),
          shapeFunctionSet_( std::move( shapeFunctionSet ) ),
          interpolation_(interpolation),
          valueProjections_( valueProjections ),
          jacobianProjections_( jacobianProjections ),
          hessianProjections_( hessianProjections ),
          size_( valueProjection()[0].size() )
      {}

      using BaseType::entity;
      using BaseType::valid;
      using BaseType::type;
      using BaseType::geometry;
      using BaseType::referenceElement;


      int order () const { return shapeFunctionSet_.order(); }

      std::size_t size () const { return size_; }

      template< class Quadrature, class DofVector, class Values >
      void evaluateAll ( const Quadrature &quadrature, const DofVector &dofs, Values &values ) const
      {
        const std::size_t nop = quadrature.nop();
        for( std::size_t qp = 0; qp < nop; ++qp )
          evaluateAll( quadrature[ qp ], dofs, values[ qp ] );
      }

      template< class F, int d, class DofVector >
      void evaluateAll ( const Dune::FieldVector<F,d> &x, const DofVector &dofs, RangeType &value ) const
      {
        // needed for plotting for example: assert(0);
        value = RangeType( 0 );
        shapeFunctionSet_.evaluateEach( x, [ this, &dofs, &value ] ( std::size_t alpha, RangeType phi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              value.axpy( DomainFieldType(valueProjection()[ alpha ][ j ]*dofs[ j ]), phi_alpha );
          } );
      }
      template< class Point, class DofVector >
      void evaluateAll ( const Point &x, const DofVector &dofs, RangeType &value ) const
      {
        value = RangeType( 0 );
        if constexpr ( Point::QuadratureType::codimension == 1)
        {
          if (outsideIntersection(x)) // intersection on polygon boundary
          {
            Std::vector < Dune::DynamicMatrix<DomainFieldType> > localDofVectorMatrix(2);
            Std::vector<Std::vector<unsigned int>> mask(2,Std::vector<unsigned int>(0));
            auto edgeSF = (*interpolation_)(x.quadrature().intersection(), localDofVectorMatrix, mask,
                                            this,
                                            x.quadrature().isInside());
            edgeSF.evaluateEach(x.localPosition(), [&](std::size_t beta, RangeType psi)
            {
              if (beta < localDofVectorMatrix[0].size())
                for (std::size_t s=0; s<mask[0].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                {
                  std::size_t alpha = mask[0][s];
                  value.axpy( DomainFieldType(localDofVectorMatrix[0][beta][s]*dofs[alpha]), psi );
                }
            });
            return;
          }
        }
        {
          shapeFunctionSet_.evaluateEach( x, [ this, &dofs, &value ] ( std::size_t alpha, RangeType phi_alpha ) {
              for( std::size_t j = 0; j < size(); ++j )
                value.axpy( DomainFieldType(valueProjection()[ alpha ][ j ]*dofs[ j ]), phi_alpha );
            } );
        }
      }

      template< class F, int d, class Values >
      void evaluateAll ( const Dune::FieldVector<F,d> &x, Values &values ) const
      {
        // needed for plotting for example: assert(0);
        assert( values.size() >= size() );
        std::fill( values.begin(), values.end(), RangeType( 0 ) );
        shapeFunctionSet_.evaluateEach( x, [ this, &values ] ( std::size_t alpha, RangeType phi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              values[ j ].axpy( valueProjection()[ alpha ][ j ], phi_alpha );
          } );
      }

      template< class Point, class Values > const
      void jacValAll ( const Point &x, Values &values ) const
      {
        assert( values.size() >= size() );
        std::fill( values.begin(), values.end(), JacobianRangeType( 0 ) );
        shapeFunctionSet_.jacValEach( position(x), [ this, &values ] ( std::size_t alpha, JacobianRangeType dphi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              values[ j ].axpy( valueProjection()[ alpha ][ j ], dphi_alpha );
          } );
      }

      template< class Point, class Values >
      void evaluateAll ( const Point &x, Values &values ) const
      {
        assert( values.size() >= size() );
        std::fill( values.begin(), values.end(), RangeType( 0 ) );
        if constexpr ( Point::QuadratureType::codimension == 1)
        {
          if (outsideIntersection(x)) // intersection on polygon boundary
          {
            // return;
            Std::vector < Dune::DynamicMatrix<DomainFieldType> > localDofVectorMatrix(2);
            Std::vector<Std::vector<unsigned int>> mask(2,Std::vector<unsigned int>(0));
            auto edgeSF = (*interpolation_)(x.quadrature().intersection(), localDofVectorMatrix, mask,
                                            this,
                                            x.quadrature().isInside());
            edgeSF.evaluateEach(x.localPosition(), [&](std::size_t beta, RangeType psi)
            {
              if (beta < localDofVectorMatrix[0].size())
                for (std::size_t s=0; s<mask[0].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                {
                  std::size_t alpha = mask[0][s];
                  values[alpha].axpy( localDofVectorMatrix[0][beta][s], psi );
                }
            });
            return;
          }
        }
        {
          shapeFunctionSet_.evaluateEach( x, [ this, &values ] ( std::size_t alpha, RangeType phi_alpha ) {
              for( std::size_t j = 0; j < size(); ++j )
                values[ j ].axpy( valueProjection()[ alpha ][ j ], phi_alpha );
            } );
        }
      }

      template< class Quadrature, class DofVector, class Jacobians >
      void jacobianAll ( const Quadrature &quadrature, const DofVector &dofs, Jacobians &jacobians ) const
      {
        const std::size_t nop = quadrature.nop();
        for( std::size_t qp = 0; qp < nop; ++qp )
          jacobianAll( quadrature[ qp ], dofs, jacobians[ qp ] );
      }
      template< class F, int d, class DofVector >
      void jacobianAll ( const Dune::FieldVector<F,d> &x, const DofVector &dofs, JacobianRangeType &jacobian ) const
      {
        jacobian = JacobianRangeType( 0 );
        shapeFunctionSet_.jacobianEach( x, [ this, &dofs, &jacobian ] ( std::size_t alpha, JacobianRangeType dphi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              jacobian.axpy( DomainFieldType(jacobianProjection()[ alpha ][ j ]*dofs[ j ]), dphi_alpha );
          } );
      }
      template< class Point, class DofVector >
      void jacobianAll ( const Point &x, const DofVector &dofs, JacobianRangeType &jacobian ) const
      {
        jacobian = JacobianRangeType( 0 );

        if constexpr ( Point::QuadratureType::codimension == 1)
        {
          if (outsideIntersection(x)) // intersection on polygon boundary
          {
            Std::vector< Dune::DynamicMatrix<DomainFieldType> > localDofVectorMatrix(2);
            Std::vector< Std::vector<unsigned int> > mask(2,Std::vector<unsigned int>(0));
            auto locx = x.localPosition();
            DomainType normal = x.quadrature().intersection().unitOuterNormal(locx);
            const auto &jit = x.quadrature().intersection().geometry().jacobianInverseTransposed(locx);
            auto edgeSF = (*interpolation_)(x.quadrature().intersection(), localDofVectorMatrix, mask,
                                            this,
                                            x.quadrature().isInside());

            // first step: take normal derivative if available
            if (localDofVectorMatrix[1].size() > 0)
            {
              edgeSF.evaluateEach(locx, [&](std::size_t beta, RangeType psi)
              {
                if (beta < localDofVectorMatrix[1].size())
                {
                  JacobianRangeType dpsi(0);
                  for (std::size_t r=0;r<dimRange;++r)
                    dpsi[r].axpy(psi[r],normal);
                  for (std::size_t s=0; s<mask[1].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                  {
                    std::size_t j = mask[1][s];
                    jacobian.axpy( DomainFieldType(localDofVectorMatrix[1][beta][s]*dofs[j]), dpsi );
                  }
                }
              });
            }
            else
            {
              shapeFunctionSet_.jacobianEach( x, [ & ] ( std::size_t alpha, JacobianRangeType dphi_alpha ) {
                  JacobianRangeType dpsi(0);
                  for (std::size_t r=0;r<dimRange;++r)
                    dpsi[r].axpy(dphi_alpha[r]*normal,normal);
                  for( std::size_t j = 0; j < size(); ++j )
                    jacobian.axpy( DomainFieldType(jacobianProjection()[ alpha ][ j ]*dofs[ j ]), dpsi );
                } );
            }
            // second step: tangential derivatives
            edgeSF.jacobianEach(locx, [&](std::size_t beta, auto dhatpsi)
            {
              if (beta < localDofVectorMatrix[0].size())
              {
                // note: edge sfs in reference coordinate so apply scaling 1/|S|
                JacobianRangeType dpsi;
                for (std::size_t r=0;r<dimRange;++r)
                {
                  jit.mv(dhatpsi[r], dpsi[r]);
                  assert( std::abs(dpsi[r]*normal) < 1e-10 );
                }
                for (std::size_t s=0; s<mask[0].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                {
                  std::size_t j = mask[0][s];
                  jacobian.axpy( DomainFieldType(localDofVectorMatrix[0][beta][s]*dofs[ j ]), dpsi );
                }
              }
            });
            return;
          }
        }
        {
          shapeFunctionSet_.jacobianEach( x, [ this, &dofs, &jacobian ] ( std::size_t alpha, JacobianRangeType dphi_alpha ) {
              for( std::size_t j = 0; j < size(); ++j )
                jacobian.axpy( DomainFieldType(jacobianProjection()[ alpha ][ j ]*dofs[ j ]), dphi_alpha );
            } );
        }
      }
      template< class F, int d, class Jacobians >
      void jacobianAll ( const Dune::FieldVector<F,d> &x, Jacobians &jacobians ) const
      {
        assert( jacobians.size() >= size() );
        std::fill( jacobians.begin(), jacobians.end(), JacobianRangeType( 0 ) );
        shapeFunctionSet_.jacobianEach( x, [ this, &jacobians ] ( std::size_t alpha, JacobianRangeType dphi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              jacobians[ j ].axpy( jacobianProjection()[ alpha ][ j ], dphi_alpha );
          } );
      }
      template< class Point, class Jacobians >
      void jacobianAll ( const Point &x, Jacobians &jacobians ) const
      {
        assert( jacobians.size() >= size() );
        std::fill( jacobians.begin(), jacobians.end(), JacobianRangeType( 0 ) );
        if constexpr ( Point::QuadratureType::codimension == 1)
        {
          if (outsideIntersection(x)) // intersection on polygon boundary
          {
            Std::vector< Dune::DynamicMatrix<DomainFieldType> > localDofVectorMatrix(2);
            Std::vector< Std::vector<unsigned int> > mask(2,Std::vector<unsigned int>(0));
            auto locx = x.localPosition();
            DomainType normal = x.quadrature().intersection().unitOuterNormal(locx);
            const auto &jit = x.quadrature().intersection().geometry().jacobianInverseTransposed(locx);
            auto edgeSF = (*interpolation_)(x.quadrature().intersection(), localDofVectorMatrix, mask,
                                            this,
                                            x.quadrature().isInside());

            // first step: take normal derivative if available
            if (localDofVectorMatrix[1].size() > 0)
              edgeSF.evaluateEach(locx, [&](std::size_t beta, RangeType psi)
              {
                if (beta < localDofVectorMatrix[1].size())
                {
                  JacobianRangeType dpsi(0);
                  for (std::size_t r=0;r<dimRange;++r)
                    dpsi[r].axpy(psi[r],normal);
                  for (std::size_t s=0; s<mask[1].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                  {
                    std::size_t alpha = mask[1][s];
                    jacobians[alpha].axpy( localDofVectorMatrix[1][beta][s], dpsi );
                  }
                }
              });
            else
              shapeFunctionSet_.jacobianEach( x, [ & ] ( std::size_t alpha, JacobianRangeType dphi_alpha ) {
                  JacobianRangeType dpsi(0);
                  for (std::size_t r=0;r<dimRange;++r)
                    dpsi[r].axpy(dphi_alpha[r]*normal,normal);
                  for( std::size_t j = 0; j < size(); ++j )
                    jacobians[ j ].axpy( jacobianProjection()[ alpha ][ j ], dpsi );
                } );
            // second step: tangential derivatives
            edgeSF.jacobianEach(locx, [&](std::size_t beta, auto dhatpsi)
            {
              if (beta < localDofVectorMatrix[0].size())
              {
                // note: edge sfs in reference coordinate so apply jit - also
                // transforms to tangential derivative
                JacobianRangeType dpsi;
                for (std::size_t r=0;r<dimRange;++r)
                {
                  jit.mv(dhatpsi[r], dpsi[r]);
                  assert( std::abs(dpsi[r]*normal) < 1e-10 );
                }
                for (std::size_t s=0; s<mask[0].size(); ++s) // note that edgePhi is the transposed of the basis transform matrix
                {
                  std::size_t alpha = mask[0][s];
                  jacobians[alpha].axpy( localDofVectorMatrix[0][beta][s], dpsi );
                }
              }
            });
            return;
          }
        }
        {
          shapeFunctionSet_.jacobianEach( x, [ this, &jacobians ] ( std::size_t alpha, JacobianRangeType dphi_alpha ) {
              for( std::size_t j = 0; j < size(); ++j )
                jacobians[ j ].axpy( jacobianProjection()[ alpha ][ j ], dphi_alpha );
            } );
        }
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
        shapeFunctionSet_.hessianEach( x, [this, &dofs, &hessian ] ( std::size_t alpha, HessianRangeType d2phi_alpha ) {
            const auto &hessianProjectionAlpha = hessianProjection()[alpha];
            for( std::size_t j = 0; j < size(); ++j )
              hessian.axpy( DomainFieldType(hessianProjectionAlpha[ j ]*dofs[ j ]), d2phi_alpha );
        } );
      }
      template< class Point, class Hessians >
      void hessianAll ( const Point &x, Hessians &hessians ) const
      {
        assert( hessians.size() >= size() );
        std::fill( hessians.begin(), hessians.end(), HessianRangeType( 0 ) );
        shapeFunctionSet_.hessianEach( x, [ this, &hessians ] ( std::size_t alpha, HessianRangeType d2phi_alpha ) {
            const auto &hessianProjectionAlpha = hessianProjection()[alpha];
            for( std::size_t j = 0; j < size(); ++j )
              hessians[ j ].axpy( hessianProjectionAlpha[ j ], d2phi_alpha );
        } );
      }

      /********************************************/

      template< class Quadrature, class Vector, class DofVector >
      void axpy ( const Quadrature &quad, const Vector &values, DofVector &dofs ) const
      {
        const unsigned int nop = quad.nop();
        for( unsigned int qp = 0; qp < nop; ++qp )
          axpy( quad[ qp ], values[ qp ], dofs );
      }
      template< class Quadrature, class VectorA, class VectorB, class DofVector >
      void axpy ( const Quadrature &quad, const VectorA &valuesA, const VectorB &valuesB, DofVector &dofs ) const
      {
        const unsigned int nop = quad.nop();
        for( unsigned int qp = 0; qp < nop; ++qp )
        {
          axpy( quad[ qp ], valuesA[ qp ], dofs );
          axpy( quad[ qp ], valuesB[ qp ], dofs );
        }
      }
      template< class Point, class DofVector >
      void_t<typename Point::QuadratureType> axpy ( const Point &x, const RangeType &valueFactor,
                  const JacobianRangeType &jacobianFactor,
                  DofVector &dofs ) const
      {
        axpy( x, valueFactor, dofs );
        axpy( x, jacobianFactor, dofs );
      }
      template< class Point, class DofVector >
      void_t<typename Point::QuadratureType> axpy ( const Point &x, const RangeType &valueFactor, DofVector &dofs ) const
      {
        std::size_t size = size_;
        assert( size == dofs.size() );
        std::vector< RangeType > values( size );
        evaluateAll(x,values);
        for (std::size_t i=0; i<size; ++i)
          dofs[i] += values[i]*valueFactor;
      }
      template< class Point, class DofVector >
      void_t<typename Point::QuadratureType> axpy ( const Point &x, const JacobianRangeType &jacobianFactor, DofVector &dofs ) const
      {
        std::size_t size = size_;
        assert( size == dofs.size() );
        std::vector< JacobianRangeType > jacobians( size );
        jacobianAll(x, jacobians);
        for (std::size_t i=0; i<size; ++i)
          for (std::size_t r=0; r<RangeType::dimension; ++r)
            dofs[i] += jacobians[i][r]*jacobianFactor[r];
      }
      template< class Point, class DofVector >
      void_t<typename Point::QuadratureType> axpy ( const Point &x, const HessianRangeType &hessianFactor, DofVector &dofs ) const
      {
        std::size_t size = size_;
        assert( size == dofs.size() );
        std::vector< HessianRangeType > hessians( size );
        hessianAll(x, hessians);
        for (std::size_t i=0; i<size; ++i)
          for (std::size_t r=0; r<RangeType::dimension; ++r)
            for (std::size_t d=0; d<DomainType::dimension; ++d)
              dofs[i] += hessians[i][r][d]*hessianFactor[r][d];
      }

    private:
      template< class Point >
      DomainType position ( const Point &x ) const
      {
        return Fem::coordinate(x);
      }
      template <class Point>
      bool outsideIntersection(const Point& x) const
      {
        assert( Point::QuadratureType::codimension == 1);
        int edgeNumber = x.quadrature().isInside()?
                         x.quadrature().intersection().indexInInside():
                         x.quadrature().intersection().indexInOutside();
        const int k = (*interpolation_).indexSet().localIndex( entity(), edgeNumber, dimDomain-1 );
        return k>=0;
      }

      std::size_t agglomerate_;
      ShapeFunctionSet shapeFunctionSet_;
      std::shared_ptr<InterpolationType> interpolation_;
      std::shared_ptr<Vector<ValueProjection>> valueProjections_;
      std::shared_ptr<Vector<JacobianProjection>> jacobianProjections_;
      std::shared_ptr<Vector<HessianProjection>> hessianProjections_;
      size_t size_;
    };

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_BASISFUNCTIONSET_HH
