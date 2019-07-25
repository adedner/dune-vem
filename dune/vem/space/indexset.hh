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

    public:
      // !TS
      typedef std::array<std::vector<int>,BaseType::dimension+1> TestSpacesType;
      typedef GridPart GridPartType;

      typedef typename BaseType::AgglomerationType AgglomerationType;
      typedef typename BaseType::AllocatorType AllocatorType;

      // !TS assume vector of vectors
      explicit VemAgglomerationIndexSet ( const AgglomerationType &agglomeration,
          const TestSpacesType &testSpaces,
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
      // !TS change to take into account vector of vector storage
      std::vector< std::pair< int, unsigned int > > dofsPerCodim () const
      {
          //issue with entries being -1 here?
        const int dimension = BaseType::dimension;
        const int vSize = 2*sumTestSpaces(0)>=0? 1:0;
        const int eSize = sumTestSpaces(1)>=0? Dune::Fem::OrthonormalShapeFunctions< GridPartType::dimension-1 >::size( testSpaces_[1][0] ) : 0;
        const int iSize = sumTestSpaces(2)>=0? Dune::Fem::OrthonormalShapeFunctions< GridPartType::dimension >::size( testSpaces_[2][0] ) : 0;
        return { std::make_pair( dimension,   vSize ),
                 std::make_pair( dimension-1, eSize ),
                 std::make_pair( dimension-2, iSize ) };
      }

      // !TS
      const std::vector<int> testSpaces() const
      {
        return {testSpaces_[0][0],testSpaces_[1][0],testSpaces_[2][0]};
      }
      std::vector<int> orders()
      {
          std::vector<int> ret(3,0);
          ret[0] += testSpaces_[2][0];
          ret[1] += testSpaces_[2][0] + 1;
          ret[2] += testSpaces_[2][0] + 2;
          return ret;
      }

      const std::vector<int> maxDegreePerCodim() const
        {
          std::vector<int> ret(3,0);
            for ( int k = 0; k < ret.size(); k++){
              ret[k] += *std::max_element( testSpaces_[k].begin(), testSpaces_[k].end() );
            }
           return ret;
      }

      std::vector<int> vertexOrders()
      {
          return testSpaces_[0];
      }

      std::vector<int> edgeOrders()
      {
          return testSpaces_[1];
      }

      std::vector<int> innerOrders()
      {
          return testSpaces_[2];
      }

    private:
      int sumTestSpaces(unsigned int codim) const
      {
        return std::accumulate(testSpaces_[codim].begin(),testSpaces_[codim].end(),0);
      }
      // !TS
      const TestSpacesType testSpaces_;
    };


  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_INDEXSET_HH
