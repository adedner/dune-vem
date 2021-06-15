#ifndef DUNE_VEM_AGGLOMERATION_DGMAPPER_HH
#define DUNE_VEM_AGGLOMERATION_DGMAPPER_HH

#include <dune/fem/space/mapper/dofmapper.hh>
#include <dune/vem/agglomeration/agglomeration.hh>

namespace Dune
{

  namespace Vem
  {

    // Internal Forward Declarations
    // -----------------------------

    template< class GridPart >
    class AgglomerationDGMapper;



    // AgglomerationDGMapperTraits
    // ---------------------------

    template< class GridPart >
    struct AgglomerationDGMapperTraits
    {
      typedef AgglomerationDGMapper< GridPart > DofMapperType;

      typedef typename GridPart::template Codim< 0 >::EntityType ElementType;

      typedef std::size_t SizeType;
    };



    // AgglomerationDGMapper
    // ---------------------

    template< class GridPart >
    class AgglomerationDGMapper
      : public Fem::DofMapper< AgglomerationDGMapperTraits< GridPart > >
    {
      typedef AgglomerationDGMapper< GridPart > ThisType;
      typedef Fem::DofMapper< AgglomerationDGMapperTraits< GridPart > > BaseType;

    public:
      typedef Agglomeration< GridPart > AgglomerationType;

      typedef typename BaseType::ElementType ElementType;

      typedef typename BaseType::SizeType SizeType;

      typedef SizeType GlobalKeyType;

      explicit AgglomerationDGMapper ( AgglomerationType &agglomeration )
        : agglomeration_( agglomeration )
      {}

      SizeType size () const noexcept { return agglomeration().size(); }

      static constexpr bool contains ( int codim ) noexcept { return (codim == 0); }

      static constexpr bool fixedDataSize ( int codim ) noexcept { return true; }

      template< class Function >
      void mapEach ( const ElementType &element, Function function ) const
      {
        function( 0, agglomeration().index( element ) );
      }

      template< class Function >
      void mapEachEntityDof ( const ElementType &element, Function function ) const
      {
        // Hack: Report all DoFs to be attached to each element for communication
        //       As DoFs will be copied, we just do too much work, here
        function( 0, agglomeration().index( element ) );
      }

      template< class Entity, class Function >
      void mapEachEntityDof ( const Entity &, Function ) const
      {
        assert( Entity::codimension != 0 );
      }

      static constexpr SizeType maxNumDofs () noexcept { return 1; }

      static constexpr SizeType numDofs ( const ElementType & ) noexcept { return 1; }

      template< class Entity >
      static constexpr SizeType numEntityDofs ( const Entity & ) noexcept
      {
        return (Entity::codimension == 0 ? 1 : 0);
      }

      // compatibility: methods deprecated for non-adaptive mappers

      static constexpr bool consecutive () noexcept { return false; }

      static constexpr SizeType numBlocks () noexcept { return 1; }

      static constexpr SizeType numberOfHoles ( int ) noexcept { return 0; }

      static SizeType oldIndex ( int, int ) { DUNE_THROW( RangeError, "AgglomerationDGMapper has no holes." ); }
      static SizeType newIndex ( int, int ) { DUNE_THROW( RangeError, "AgglomerationDGMapper has no holes." ); }

      static constexpr SizeType oldOffSet ( int ) noexcept { return 0; }
      static constexpr SizeType offSet ( int ) noexcept { return 0; }

      // implementation-defined methods

      const AgglomerationType &agglomeration () const { return agglomeration_; }
      AgglomerationType &agglomeration () { return agglomeration_; }

    private:
      AgglomerationType &agglomeration_;
    };

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_DGMAPPER_HH
