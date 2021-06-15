#pragma once
namespace Dune
{
  namespace Vem
  {
    namespace Std
    {
      template <class T>
      struct vector : public std::vector<T>
      {
        using std::vector<T>::vector;
        vector(const std::vector<T> &other) : std::vector<T>(other) {}
        T &operator[](std::size_t i) {
          assert(i<this->size()); return std::vector<T>::operator[](i);
        }
        const T &operator[](std::size_t i) const {
          assert(i<this->size()); return std::vector<T>::operator[](i);
        }
      };
    }
  }
  template<class T>
  struct FieldTraits< Vem::Std::vector<T> >
  {
    typedef typename FieldTraits<T>::field_type field_type;
    typedef typename FieldTraits<T>::real_type real_type;
  };
}
