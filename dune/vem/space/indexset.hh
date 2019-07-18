#ifndef DUNE_VEM_SPACE_INDEXSET_HH
#define DUNE_VEM_SPACE_INDEXSET_HH

#include <cassert>
#include <cstddef>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include <dune/grid/common/rangegenerators.hh>

#include <dune/vem/agglomeration/agglomeration.hh>
#include <dune/vem/agglomeration/boundingbox.hh>
#include <dune/vem/agglomeration/indexset.hh>

namespace Dune
{

  namespace Vem
  {

    // AgglomerationIndexSet
    // ---------------------

    template< class GridPart, class Allocator = std::allocator< std::size_t > >
    class VemAgglomerationIndexSet
    : public AgglomerationIndexSet< GridPart, Allocator >
    {
      typedef VemAgglomerationIndexSet< GridPart, Allocator > ThisType;
      typedef AgglomerationIndexSet< GridPart, Allocator > BaseType;

      std::vector <int> testSpaces_;
    public:
      typedef GridPart GridPartType;

      typedef typename BaseType::AgglomerationType AgglomerationType;
      typedef typename BaseType::AllocatorType AllocatorType;

      explicit VemAgglomerationIndexSet ( const AgglomerationType &agglomeration,
          std::vector<int> testSpaces,
          AllocatorType allocator = AllocatorType() )
      : BaseType( agglomeration, allocator )
      , testSpaces_( testSpaces )
      {
        /*
        std::cout << "####################\n";
        std::cout << testSpaces_[0] << " " << dofsPerCodim()[0].second << std::endl;
        std::cout << testSpaces_[1] << " " << dofsPerCodim()[1].second << std::endl;
        std::cout << testSpaces_[2] << " " << dofsPerCodim()[2].second << std::endl;
        std::cout << "####################\n";
        */
      }

      // return the number of dofs per codimension
      std::vector< std::pair< int, unsigned int > > dofsPerCodim () const
      {
        const int dimension = BaseType::dimension;
        const int vSize = testSpaces_[0]>=0? 1:0;
        const int eSize = testSpaces_[1]>=0? Dune::Fem::OrthonormalShapeFunctions< GridPartType::dimension-1 >::size( testSpaces_[1] ) : 0;
        const int iSize = testSpaces_[2]>=0? Dune::Fem::OrthonormalShapeFunctions< GridPartType::dimension >::size( testSpaces_[2] ) : 0;
        return { std::make_pair( dimension,   vSize ),
                 std::make_pair( dimension-1, eSize ),
                 std::make_pair( dimension-2, iSize ) };
      }

      const std::vector<int> &testSpaces() const
      {
        return testSpaces_;
      }

    };


  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_INDEXSET_HH
