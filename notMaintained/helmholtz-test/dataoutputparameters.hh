// include parameter handling
#include <dune/fem/io/parameter.hh>

// DataOutputParameters
// --------------------

struct DataOutputParameters
: public Dune::Fem::LocalParameter< Dune::Fem::DataOutputParameters, DataOutputParameters >
{
DataOutputParameters ( const int step )
: step_( step )
{}

  DataOutputParameters ( const DataOutputParameters &other )
    : step_( other.step_ )
  {}

  std::string prefix () const
  {
    std::stringstream s;
    s << "poisson-" << step_ << "-";
    return s.str();
  }

  private:
  int step_;
};
