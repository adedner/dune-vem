#ifndef DUNE_VEM_AGGLOMERATION_FUNCTOR_HH
#define DUNE_VEM_AGGLOMERATION_FUNCTOR_HH

#include <utility>

namespace Dune
{

  namespace Vem
  {

    // TransformedAssign
    // -----------------

    template< class Transformation >
    struct TransformedAssign
    {
      TransformedAssign ( Transformation transformation )
        : transformation_( std::move( transformation ) )
      {}

      template< class A, class B >
      void operator() ( const A &a, B &b ) const
      {
        b = transformation_( a );
      }

    private:
      Transformation transformation_;
    };

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_FUNCTOR_HH
