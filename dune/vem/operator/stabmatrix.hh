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

namespace Dune
{
  namespace Vem
  {
    template< class LinOperator >
    void stabilization(LinOperator &op,
         // std::optional<double> hessStabilization, std::optional<double> gradStabilization, std::optional<double> massStabilization)
         double hessStab, double gradStab, double massStab)
    {
      typedef typename LinOperator::DomainFunctionType DomainFunctionType;
      typedef typename LinOperator::RangeFunctionType RangeFunctionType;
      typedef typename RangeFunctionType::DiscreteFunctionSpaceType DiscreteFunctionSpaceType;
      typedef typename DiscreteFunctionSpaceType::GridPartType GridPartType;
      typedef typename GridPartType::template Codim< 0 >::EntitySeedType ElementSeedType;

      // double hessStab = hessStabilization.value_or(0.);
      // double gradStab = gradStabilization.value_or(1.);
      // double massStab = massStabilization.value_or(0.);

      const DiscreteFunctionSpaceType &domainSpace = op.domainSpace();
      const DiscreteFunctionSpaceType &rangeSpace = op.rangeSpace();
      const int domainBlockSize = domainSpace.localBlockSize;
      const GridPartType &gridPart = rangeSpace.gridPart();
      const auto &agIndexSet    = rangeSpace.blockMapper().indexSet();
      const auto &agglomeration = rangeSpace.agglomeration();

      Fem::DiagonalStencil< DiscreteFunctionSpaceType, DiscreteFunctionSpaceType >
           stencil(domainSpace,rangeSpace);
      op.reserve( stencil );
      op.clear();
      typedef Dune::Fem::TemporaryLocalMatrix< DiscreteFunctionSpaceType,
                                               DiscreteFunctionSpaceType > TemporaryLocalMatrixType;
      TemporaryLocalMatrixType jLocal( domainSpace, rangeSpace );
      for (const auto &entity : Dune::elements(gridPart, Dune::Partitions::interiorBorder))
      {
        const std::size_t agglomerate = agglomeration.index( entity );
        const auto &bbox = agIndexSet.boundingBox( agglomerate );
        double bbH2 = pow(bbox.volume()/bbox.diameter(),2);
        const auto &stabMatrix = rangeSpace.stabilization(entity);
        jLocal.init( entity, entity );
        jLocal.clear();
        std::size_t bs = domainBlockSize;
        assert( jLocal.rows()    == stabMatrix.rows()*bs );
        assert( jLocal.columns() == stabMatrix.cols()*bs );
        assert( stabMatrix.cols()*bs == uLocal.size() );

        auto stab = gradStab + massStab*bbH2 + hessStab/bbH2;

        for (std::size_t r = 0; r < stabMatrix.rows(); ++r)
          for (std::size_t c = 0; c < stabMatrix.cols(); ++c)
            for (std::size_t b = 0; b < bs; ++b)
              jLocal.add(r*bs+b, c*bs+b, stab*stabMatrix[r][c]);
        op.addLocalMatrix( entity, entity, jLocal );
      }
      op.flushAssembly();
    }

  } // namespace Vem
} // namespace Dune

#endif // #ifndef DUNE_VEM_OPERATOR_STABMATRIX_HH
