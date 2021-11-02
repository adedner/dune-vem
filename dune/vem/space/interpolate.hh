#ifndef DUNE_VEM_SPACE_INTERPOLATE_HH
#define DUNE_VEM_SPACE_INTERPOLATE_HH

#include <type_traits>
#include <vector>

#include <dune/grid/common/partitionset.hh>
#include <dune/grid/common/rangegenerators.hh>

#include <dune/fem/function/common/discretefunction.hh>
#include <dune/vem/misc/vector.hh>

namespace Dune
{

  namespace Vem
  {

    // External Forward Declarations
    // -----------------------------

    template< class DiscreteFunctionSpace >
    struct IsAgglomerationVEMSpace;

  } // namespace Vem



  namespace Fem
  {

    // interpolate
    // -----------

    template< class GridFunction, class DiscreteFunction, unsigned int partitions >
    static inline std::enable_if_t< std::is_convertible< GridFunction, HasLocalFunction >::value && Dune::Vem::IsAgglomerationVEMSpace< typename DiscreteFunction::DiscreteFunctionSpaceType >::value >
    interpolate ( const GridFunction &u, DiscreteFunction &v, PartitionSet< partitions > ps )
    {
      const auto &mapper = v.space().blockMapper();
      const auto &agglomeration = mapper.agglomeration();
      const int blockSize = DiscreteFunction::DiscreteFunctionSpaceType::localBlockSize;

      // reserve memory for local dof vector
      std::vector< typename DiscreteFunction::RangeFieldType > ldv;
      ldv.reserve( mapper.maxNumDofs() * blockSize );

      typedef typename DiscreteFunction::GridPartType GridPartType;
      typedef typename GridPartType::template Codim< 0 >::EntitySeedType ElementSeedType;
      Dune::Vem::Std::vector< Dune::Vem::Std::vector< ElementSeedType > > entitySeeds( agglomeration.size() );
      for( const auto &element : elements( v.gridPart(), ps ) )
        entitySeeds[ agglomeration.index( element ) ].push_back( element.seed() );

      const auto& interpolation = v.space().interpolation();

      Dune::Fem::ConstLocalFunction<GridFunction> uLocal(u);
      for( std::size_t agglomerate = 0; agglomerate < agglomeration.size(); ++agglomerate )
      {
        if( entitySeeds[ agglomerate ].empty() )
          continue;

        ldv.resize( mapper.numDofs( agglomerate ) * blockSize );
        std::fill(ldv.begin(),ldv.end(),0);
        for( const ElementSeedType &entitySeed : entitySeeds[ agglomerate ] )
        {
          const auto &element = v.gridPart().entity( entitySeed );
          uLocal.bind(element);
          interpolation( element, uLocal, ldv );
        }
        v.setLocalDofs( v.gridPart().entity( entitySeeds[ agglomerate ].front() ), ldv );
      }
    }

  } // namespace Fem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_INTERPOLATE_HH
