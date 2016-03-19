#ifndef DUNE_VEM_OPERATOR_CONSTRAINTS_NOCONSTRAINTS_HH
#define DUNE_VEM_OPERATOR_CONSTRAINTS_NOCONSTRAINTS_HH

namespace Dune
{

  namespace Vem
  {

    struct NoConstraints
    {
      template< class Model, class DiscreteFunctionSpace >
      NoConstraints ( const Model &, const DiscreteFunctionSpace & )
      {}

      template< class DiscreteFunction >
      void operator() ( const DiscreteFunction &u, DiscreteFunction &w ) const
      {}

      template< class GridFunction, class DiscreteFunction >
      void operator() ( const GridFunction &u, DiscreteFunction &w ) const
      {}

      template< class LinearOperator >
      void applyToOperator ( LinearOperator &linearOperator ) const
      {}
    };

  } // namespace Vem

} // namespace DUne

#endif // #ifndef DUNE_VEM_OPERATOR_CONSTRAINTS_NOCONSTRAINTS_HH
