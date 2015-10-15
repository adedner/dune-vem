#ifndef DUNE_VEM_FUNCTION_SIMPLE_HH
#define DUNE_VEM_FUNCTION_SIMPLE_HH

#include <utility>

#include <dune/fem/space/common/functionspace.hh>

namespace Dune
{

  namespace Vem
  {

    // SimpleFunctionTraits
    // --------------------

    template< class Domain, class Function, class Range = typename std::result_of< Function( const Domain & ) >::type >
    struct SimpleFunctionTraits;

    template< class DF, int dimD, class Function, class RF, int dimR >
    struct SimpleFunctionTraits< FieldVector< DF, dimD >, Function, FieldVector< RF, dimR > >
    {
      typedef Fem::FunctionSpace< DF, RF, dimD, dimR > FunctionSpaceType;
    };



    // SimpleFunction
    // --------------

    template< class Domain, class Function >
    class SimpleFunction
    {
      typedef SimpleFunction< Domain, Function > ThisType;

    public:
      typedef typename SimpleFunctionTraits< Domain, Function >::FunctionSpaceType FunctionSpaceType;

      typedef typename FunctionSpaceType::DomainType DomainType;
      typedef typename FunctionSpaceType::RangeType RangeType;

      explicit SimpleFunction ( Function function ) : function_( std::move( function ) ) {}

      void evaluate ( const DomainType &x, RangeType &value ) const { value = function_( x ); }

    private:
      Function function_;
    };



    // simpleFunction
    // --------------

    template< class Domain, class Function >
    inline static SimpleFunction< Domain, Function > simpleFunction ( Function function )
    {
      return SimpleFunction< Domain, Function >( std::move( function ) );
    }

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_FUNCTION_SIMPLE_HH
