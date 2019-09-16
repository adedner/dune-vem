#ifndef DUNE_FEMPY_PY_VEMINTEGRANDS_HH
#define DUNE_FEMPY_PY_VEMINTEGRANDS_HH

#include <dune/vem/operator/diffusionmodel.hh>
#include <dune/fempy/py/integrandsbase.hh>

namespace Dune
{

  namespace FemPy
  {
    template< class Integrands, class... options >
    inline void registerIntegrands ( pybind11::handle scope,
        pybind11::class_<Integrands,options...> cls)
    {
      typedef typename Integrands::GridPartType GridPart;
      typedef typename Integrands::DomainValueType DomainValue;
      typedef typename Integrands::RangeValueType RangeValue;
      typedef Fem::VirtualizedVemIntegrands< GridPart, DomainValue, RangeValue > VirtualizedIntegrands;

      detail::registerIntegrands( scope, cls );

      detail::clsVirtualizedIntegrands< VirtualizedIntegrands >( scope ).
        def( pybind11::init( [] ( Integrands &integrands ) {
          return new VirtualizedIntegrands( std::ref( integrands ) );
        }), pybind11::keep_alive< 1, 2 >() );
      pybind11::implicitly_convertible< Integrands, VirtualizedIntegrands >();
    }

  }
}

#endif // DUNE_FEMPY_PY_INTEGRANDS_HH
