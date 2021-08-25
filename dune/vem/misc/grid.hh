#include <dune/alugrid/grid.hh>
#include <dune/alugrid/dgf.hh>
#include <dune/grid/yaspgrid.hh>
namespace Dune
{
  namespace Vem
  {
    template <int dg=2, int dw=2>
    using Grid = Dune::ALUGrid< dg, dw, Dune::simplex, Dune::conforming >;
    template <int dg=2, int dw=2>
    using CubeGrid = Dune::ALUGrid< dg, dw, Dune::cube, Dune::nonconforming >;
    template <int dg, class Coordinates>
    using YGrid = Dune::YaspGrid< dg, Coordinates >;
  }
}
