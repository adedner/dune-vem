#ifndef DUNE_VEM_OPERATOR_STABMATRIX_HH
#define DUNE_VEM_OPERATOR_STABMATRIX_HH

#include <utility>
#include <vector>

#include <dune/grid/common/rangegenerators.hh>
#include <dune/fem/operator/common/stencil.hh>
#include <dune/fem/operator/common/operator.hh>
#include <dune/fem/function/common/localcontribution.hh>
#include <dune/fem/common/bindguard.hh>
#include <dune/fem/operator/common/temporarylocalmatrix.hh>
#include <dune/fem/quadrature/cachingquadrature.hh>

namespace Dune
{
  namespace Vem
  {
    template< class LinOperatorS, class DSpace, class RSpace, class LinOperator>
    void stabilization(const LinOperatorS &op,
                       const DSpace &domainSpace, const RSpace &rangeSpace,
                       double hessStab, double gradStab, double massStab,
                       LinOperator &out)
    {
      typedef typename LinOperator::DomainFunctionType DomainFunctionType;
      typedef typename LinOperator::RangeFunctionType RangeFunctionType;
      typedef typename RangeFunctionType::DiscreteFunctionSpaceType DiscreteFunctionSpaceType;
      typedef typename DiscreteFunctionSpaceType::RangeType RangeType;
      typedef typename DiscreteFunctionSpaceType::JacobianRangeType JacobianRangeType;
      typedef typename DiscreteFunctionSpaceType::HessianRangeType HessianRangeType;
      typedef typename DiscreteFunctionSpaceType::GridPartType GridPartType;
      typedef typename GridPartType::template Codim< 0 >::EntitySeedType ElementSeedType;

      const int domainBlockSize = domainSpace.localBlockSize;
      const GridPartType &gridPart = rangeSpace.gridPart();
      const auto &agIndexSet    = rangeSpace.blockMapper().indexSet();
      const auto &agglomeration = rangeSpace.agglomeration();

      typedef Dune::Fem::TemporaryLocalMatrix< DiscreteFunctionSpaceType,
                                               DiscreteFunctionSpaceType > TemporaryLocalMatrixType;
      TemporaryLocalMatrixType jLocal( domainSpace, rangeSpace );
      TemporaryLocalMatrixType diagLocal( domainSpace, rangeSpace );

      ////////////////////////
      std::vector< ElementSeedType > seeds( agglomeration.size() );
      std::vector< std::vector<double> > diagLocal0( agglomeration.size() );
      std::vector< std::vector<double> > diagLocal1( agglomeration.size() );
      std::vector< std::vector<double> > diagLocal2( agglomeration.size() );
      typedef Dune::Fem::ElementQuadrature< GridPartType, 0, Dune::FemPy::FempyQuadratureTraits > QuadratureType;
      for (const auto &entity : Dune::elements(gridPart, Dune::Partitions::interiorBorder))
      {
        const unsigned int agglomerate = agglomeration.index(entity);
        const auto &baseSet = domainSpace.basisFunctionSet(entity);
        const std::size_t numDofs = baseSet.size();
        if (!seeds[ agglomerate ].isValid())
        {
          seeds[ agglomerate ] = entity.seed();
          diagLocal0[ agglomerate ].resize(numDofs,0);
          diagLocal1[ agglomerate ].resize(numDofs,0);
          diagLocal2[ agglomerate ].resize(numDofs,0);
        }
        std::vector< RangeType > phi( numDofs );
        std::vector< JacobianRangeType > dphi( numDofs );
        std::vector< HessianRangeType > d2phi( numDofs );
        QuadratureType quadrature( entity, 2*domainSpace.order() );
        const auto &geometry = entity.geometry();
        const size_t numQuadraturePoints = quadrature.nop();
        for( size_t pt = 0; pt < numQuadraturePoints; ++pt )
        {
          const typename QuadratureType::CoordinateType &x = quadrature.point( pt );
          const double weight = quadrature.weight( pt ) * geometry.integrationElement( x );
          baseSet.evaluateAll( quadrature[ pt ], phi );
          baseSet.jacobianAll( quadrature[ pt ], dphi );
          baseSet.hessianAll( quadrature[ pt ], d2phi );
          for ( size_t i=0;i<numDofs;++i)
          {
            for (size_t r=0;r<RangeType::dimension;++r)
            {
              diagLocal0[ agglomerate ][i] += phi[i][r]*phi[i][r] * weight;
              diagLocal1[ agglomerate ][i] += dphi[i][r]*dphi[i][r] * weight;
              for (size_t c=0;c<d2phi[i][r].size();++c)
                diagLocal2[ agglomerate ][i] += d2phi[i][r][c][c]*d2phi[i][r][c][c] * weight;
            }
          }
        }
      }
      ////////////////////////

      for (const auto &seed : seeds)
      {
        const auto entity = gridPart.entity( seed );
        const std::size_t agglomerate = agglomeration.index( entity );
        const auto &bbox = agIndexSet.boundingBox( agglomerate );
        double bbH2 = pow(bbox.volume()/bbox.diameter(),2);
        const auto &stabMatrix = rangeSpace.stabilization(entity);
        jLocal.init( entity, entity );
        jLocal.clear();
        std::size_t bs = domainBlockSize;
        assert( jLocal.rows()    == stabMatrix.rows()*bs );
        assert( jLocal.columns() == stabMatrix.cols()*bs );

        // auto stab = gradStab + massStab*bbH2 + hessStab/bbH2;
        std::vector<double> sRow(stabMatrix.cols());
        for (std::size_t r = 0; r < stabMatrix.rows(); ++r)
        {
          for (std::size_t c = 0; c < stabMatrix.cols(); ++c)
          {
            sRow[c] = 0;
            for (std::size_t k = 0; k < stabMatrix.cols(); ++k)
            {
              double fac = massStab*std::max(diagLocal0[agglomerate][k], bbH2) +
                           gradStab*std::max(diagLocal1[agglomerate][k], gradStab) +
                           hessStab*std::max(diagLocal2[agglomerate][k], 1/bbH2);
              sRow[c] += stabMatrix[k][r] * stabMatrix[k][c] * fac;
            }
          }
          for (std::size_t c = 0; c < stabMatrix.cols(); ++c)
            for (std::size_t b = 0; b < bs; ++b)
              jLocal.add(r*bs+b, c*bs+b, sRow[c]);
        }
        out.addLocalMatrix( entity, entity, jLocal );
      }
      out.flushAssembly();
    }

    template< class LinOperator >
    void stabilization(LinOperator &op,
         double hessStab, double gradStab, double massStab)
    {
      typedef typename LinOperator::DomainFunctionType DomainFunctionType;
      typedef typename LinOperator::RangeFunctionType RangeFunctionType;
      typedef typename RangeFunctionType::DiscreteFunctionSpaceType DiscreteFunctionSpaceType;
      Fem::DiagonalStencil< DiscreteFunctionSpaceType, DiscreteFunctionSpaceType >
           stencil(op.domainSpace(),op.rangeSpace());
      op.reserve( stencil );
      op.clear();
      stabilization(op,op.domainSpace(),op.rangeSpace(),
                    hessStab,gradStab,massStab,op);
    }

  } // namespace Vem
} // namespace Dune

#endif // #ifndef DUNE_VEM_OPERATOR_STABMATRIX_HH
