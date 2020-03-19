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
#if 0
        std::cout << "######## dofs per codim ############\n";
        std::cout << testSpaces_[0][0] << " " << dofsPerCodim()[0].second << std::endl;
        std::cout << testSpaces_[1][0] << " " << dofsPerCodim()[1].second << std::endl;
        std::cout << testSpaces_[2][0] << " " << dofsPerCodim()[2].second << std::endl;
        std::cout << "--------- orders for derivatives ---\n";
        std::cout << orders()[0] << " " << orders()[1] << " " << orders()[2] << std::endl;
        std::cout << "----------edge orders --------------\n";
        std::cout << edgeDegrees()[0] << " " << edgeDegrees()[1] << std::endl;
        std::cout << "####################\n";
#endif
      }

      // return the number of dofs per codimension
      // !TS change to take into account vector of vector storage
      std::vector< std::pair< int, unsigned int > > dofsPerCodim () const
      {
        const int dimension = BaseType::dimension;
        int vSize = 0;
        int eSize = 0;
        int iSize = 0;
        for (size_t i=0;i<testSpaces_[0].size();++i) // order2size fails for dim=dimension
          vSize += (testSpaces_[0][i]>=0) ? pow(BaseType::dimension,i):0;
        for (size_t i=0;i<testSpaces_[1].size();++i)
          eSize += order2size<1>(i);
        for (size_t i=0;i<testSpaces_[2].size();++i)
          iSize += order2size<2>(i);
        return { std::make_pair( dimension,   vSize ),
                 std::make_pair( dimension-1, eSize ),
                 std::make_pair( dimension-2, iSize ) };
      }

      std::vector<int> orders()
      {
          std::vector<int> ret(3,0);
          ret[0] += testSpaces_[2][0];
          ret[1] += std::min( {testSpaces_[2][0] + 1, edgeDegrees()[0]} );
          ret[2] += std::min( {testSpaces_[2][0] + 2, edgeDegrees()[0]+1, edgeDegrees()[1]} );
          return ret;
      }

      const std::vector<int> maxDegreePerCodim() const
        {
          std::vector<int> ret(3);
            for ( int k = 0; k < ret.size(); k++){
              ret[k] = *std::max_element( testSpaces_[k].begin(), testSpaces_[k].end() );
            }
           return ret;
      }

      std::vector<int> edgeDegrees() const
      {
        assert( testSpaces_[0].size()<2 );
        std::vector<int> degrees(2, -1);
        for (std::size_t i=0;i<testSpaces_[0].size();++i)
          if (i<testSpaces_[0].size())
            degrees[i] += 2*(testSpaces_[0][i]+1);
        for (std::size_t i=0;i<testSpaces_[1].size();++i)
          if (i<testSpaces_[1].size())
            degrees[i] += std::max(0,testSpaces_[1][i]+1);
        return degrees;
      }
      std::size_t edgeSize(int deriv) const
      {
        auto degrees = edgeDegrees();
        return degrees[deriv] < 0 ? 0 :
              Dune::Fem::OrthonormalShapeFunctions<1>::size( degrees[deriv] );
      }
      std::size_t maxEdgeDegree() const
      {
        auto degrees = edgeDegrees();
        return *std::max_element(degrees.begin(),degrees.end());
      }

      const std::vector<int> vertexOrders() const
      {
          return testSpaces_[0];
      }

      const std::vector<int> edgeOrders() const
      {
          return testSpaces_[1];
      }

      const std::vector<int> innerOrders() const
      {
          return testSpaces_[2];
      }

      template <int dim>
      std::size_t order2size(unsigned int deriv) const
      {
        if (testSpaces_[dim].size()<=deriv || testSpaces_[dim][deriv]<0)
          return 0;
        else
        {
          if (dim>0)
            return Dune::Fem::OrthonormalShapeFunctions<dim>::
              size(testSpaces_[dim][deriv]);
          else
            return pow(BaseType::dimension,deriv);
        }
      }


    private:
      std::size_t sumTestSpaces(unsigned int codim) const
      {
        return std::accumulate(testSpaces_[codim].begin(),testSpaces_[codim].end(),0);
      }
      // !TS
      const TestSpacesType testSpaces_;
    };


  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_INDEXSET_HH
