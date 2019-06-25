#include <dune/alugrid/grid.hh>
#include <dune/alugrid/dgf.hh>
namespace Dune
{
  namespace Vem
  {
    template <int dg=2, int dw=2>
    using Grid = Dune::ALUGrid< dg, dw, Dune::simplex, Dune::nonconforming >;
  }
}
