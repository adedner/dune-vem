#include <dune/geometry/referenceelements.hh>
#include <dune/fem/function/localfunction/bindable.hh>
#include <dune/vem/misc/vector.hh>

template <class GridPart, class Matrix, class SFS>
struct Derivative : public Dune::Fem::BindableGridFunction< GridPart, Dune::Dim<2> >
{
  typedef Dune::Fem::BindableGridFunction<GridPart, Dune::Dim<2> > Base;
  using Base::Base;

  Derivative(const GridPart &gridPart,
             const Matrix &matrix, const SFS &sfs, int alpha)
    : Base(gridPart)
    , alpha_(alpha)
    , matrix_(matrix)
    , sfs_(sfs)
  {}

  template <class Point>
  void evaluate(const Point &x, typename Base::RangeType &ret) const
  {
    ret = typename Base::RangeType(0.);
    sfs_.evaluateEach( x, [ & ] ( std::size_t beta, const Dune::FieldVector<double,1> &phi ) {
        if ( alpha_ < matrix_.size() &&
             beta < matrix_[alpha_].size() )
        {
          ret[0] += matrix_[beta][alpha_][0] * phi[0];
          ret[1] += matrix_[beta][alpha_][1] * phi[0];
        }
    } );
  }
  unsigned int order() const { return 2; }
  std::string name() const { return "Derivative"; }
  private:
  int alpha_;
  const Matrix &matrix_;
  const SFS &sfs_;
};
template <class GridPart, class Matrix, class SFS>
struct PhiEdge : public Dune::Fem::BindableGridFunction< GridPart, typename SFS::RangeType >
{
  static const int dimRange = SFS::RangeType::dimension;
  typedef Dune::Fem::BindableGridFunction<GridPart, typename SFS::RangeType > Base;
  using Base::Base;
  typedef typename GridPart::IntersectionType IntersectionType;

  PhiEdge(const GridPart &gridPart,
          const IntersectionType &intersection,
          const Matrix &matrix, const SFS &sfs, int i)
    : Base(gridPart)
    , intersection_(intersection)
    , i_(i)
    , matrix_(matrix)
    , sfs_(sfs)
  {}

  template <class Point>
  void evaluate(const Point &p, typename Base::RangeType &ret) const
  {
    const auto& entity = intersection_.inside();
    ret = typename Base::RangeType(0.);
    // test if evaluation point on edge
    auto x  = Dune::Fem::coordinate(p);
    auto y  = intersection_.geometryInInside().local(x);
    auto xx = intersection_.geometryInInside().global(y);
    xx -= x;
    if ( xx.two_norm() > 1e-10 ) return;
    sfs_.evaluateEach( y, [ & ] ( std::size_t beta, const typename Base::RangeType &phi ) {
        if ( beta < matrix_.size() )
        {
          assert( i_ < matrix_[beta].size() );
          for (int r=0;r<dimRange;++r)
            ret[r] += matrix_[beta][i_] * phi[r];
        }
    } );
  }
  template <class Point>
  void jacobian(const Point &p, typename Base::JacobianRangeType &ret) const
  {
    const int dimension = GridPart::dimension;
    const auto& normal = intersection_.centerUnitOuterNormal();
    const auto tau = Dune::FieldVector<double,dimension>{normal[1],-normal[0]};
    const auto& entity = intersection_.inside();
    ret = typename Base::JacobianRangeType(0.);
    // test if evaluation point on edge
    auto x  = Dune::Fem::coordinate(p);
    auto y  = intersection_.geometryInInside().local(x);
    auto xx = intersection_.geometryInInside().global(y);
    xx -= x;
    if ( xx.two_norm() > 1e-10 ) return;
    sfs_.jacobianEach( y, [ & ] ( std::size_t beta, const auto &dphi ) {
        if ( beta < matrix_.size() )
        {
          assert( i_ < matrix_[beta].size() );
          // ??????? NEEDS FIXING ret[0].axpy(matrix_[beta][i_] * dphi[0][0], tau);
        }
    } );
  }
  unsigned int order() const { return 2; }
  std::string name() const { return "Derivative"; }
  private:
  const IntersectionType &intersection_;
  int i_;
  const Matrix &matrix_;
  const SFS &sfs_;
};
