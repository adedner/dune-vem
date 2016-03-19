#ifndef DUNE_VEM_SPACE_INTERPOLATE_HH
#define DUNE_VEM_SPACE_INTERPOLATE_HH

#include <vector>

#include <dune/grid/common/partitionset.hh>
#include <dune/grid/common/rangegenerators.hh>

#include <dune/vem/space/interpolation.hh>

namespace Dune
{

  namespace Vem
  {

    // interpolate
    // -----------

    template< class GridFunction, class DiscreteFunction, unsigned int partitions >
    static inline void interpolate ( const GridFunction &u, DiscreteFunction &v, PartitionSet< partitions > ps )
    {
      const auto &mapper = v.space().blockMapper();
      const auto &agglomeration = mapper.agglomeration();

      // reserve memory for local dof vector
      std::vector< typename DiscreteFunction::RangeFieldType > ldv;
      ldv.reserve( mapper.maxNumDofs() * DiscreteFunction::DiscreteFunctionSpaceType::localBlockSize );

      typedef typename DiscreteFunction::GridPartType GridPartType;
      typedef typename GridPartType::template Codim< 0 >::EntitySeedType ElementSeedType;
      std::vector< std::vector< ElementSeedType > > entitySeeds( agglomeration.size() );
      for( const auto &element : elements( static_cast< typename GridPartType::GridViewType >( v.gridPart() ), ps ) )
        entitySeeds[ agglomeration.index( element ) ].push_back( element.seed() );

      auto interpolation = agglomerationVEMInterpolation( mapper.indexSet() );
      for( std::size_t agglomerate = 0; agglomerate < agglomeration.size(); ++agglomerate )
      {
        if( entitySeeds[ agglomerate ].empty() )
          continue;

        ldv.resize( mapper.numDofs( agglomerate ) );
        for( const ElementSeedType &entitySeed : entitySeeds[ agglomerate ] )
          interpolation( u.localFunction( v.gridPart().entity( entitySeed ) ), ldv );
        v.setLocalDofs( v.gridPart().entity( entitySeeds[ agglomerate ].front() ), ldv );
      }
    }

    template< class GridFunction, class DiscreteFunction >
    static inline void interpolate ( const GridFunction &u, DiscreteFunction &v )
    {
      interpolate( u, v, Partitions::all );
    }

  } // namespace Fem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_INTERPOLATE_HH
