#include <dune/alugrid/grid.hh>
#include <dune/alugrid/dgf.hh>
namespace Dune
{
  namespace Vem
  {
    using Grid = Dune::ALUGrid< 2, 2, Dune::simplex, Dune::nonconforming >;
  }
}
