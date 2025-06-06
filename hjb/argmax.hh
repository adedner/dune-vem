// https://optimlib.readthedocs.io/en/latest/index.html

#include <vector>
// #include <numbers>
#define M_PI 3.14159265358979

#include <dune/common/fvector.hh>
#include <dune/fem/function/localfunction/const.hh>

#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include "optim.hpp"

using optim::ColVec_t;
using optim::Mat_t;

template <class GridView, class LGF, class DLGF>
auto alphaMax(LGF &Lgf, DLGF &DLgf)
{
  static const int dimP = DLGF::FunctionSpaceType::dimRange;
  return [&](const auto& en,const auto& xLocal) -> auto
  {
    Dune::FieldVector<double,dimP> ret;
    #if 0
    auto x = en.geometry().global(xLocal);
    ret[0] = x[0]/M_PI;
    return ret;
    #endif
    // switch signs since we want to maximize
    auto objfn = [&](const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data) -> double
    {
      for (std::size_t i=0;i<dimP;++i)
      {
        Lgf.conalpha()[i] = vals_inp(i);
        DLgf.conalpha()[i] = vals_inp(i);
      }
      auto Lgf_loc = Dune::Fem::ConstLocalFunction<LGF>(en,Lgf);
      auto val = Lgf_loc.evaluate(xLocal);
      auto DLgf_loc = Dune::Fem::ConstLocalFunction<DLGF>(en,DLgf);
      auto dval = DLgf_loc.evaluate(xLocal);
      if (grad_out)
      {
        (*grad_out)(0) = -dval[0];
        (*grad_out)(1) = -dval[1];
      }
      return -val[0];
    };
    auto constrfn = [&](const ColVec_t& vals_inp, Mat_t* jacob_out, void* opt_data) -> ColVec_t
    {
      // auto pi = std::numbers::pi;
      /*
      auto pi = M_PI;
      double a1l=0, a1r=pi/3;
      double a2l=0, a2r=2*pi;

      double a1 = vals_inp(0);
      double a2 = vals_inp(1);

      ColVec_t constr_vals{{a1l-a1,a1-a1r, a2l-a2,a2-a2r}};

      if (jacob_out) {
          BMO_MATOPS_SET_SIZE_POINTER(jacob_out,4,2);

          (*jacob_out)(0,0) = -1.0;
          (*jacob_out)(0,1) =  0.0;
          (*jacob_out)(1,0) =  1.0;
          (*jacob_out)(1,1) =  0.0;
          (*jacob_out)(2,0) =  0.0;
          (*jacob_out)(2,1) = -1.0;
          (*jacob_out)(2,0) =  0.0;
          (*jacob_out)(2,1) =  1.0;
      }
      */
      double al=0, ar=1;
      double a = vals_inp(0);

      ColVec_t constr_vals{{al-a,a-ar}};

      if (jacob_out) {
          BMO_MATOPS_SET_SIZE_POINTER(jacob_out,2,1);

          (*jacob_out)(0,0) = -1.0;
          (*jacob_out)(1,0) =  1.0;
      }
      return constr_vals;
    };

    auto alpha = Eigen::VectorXd(dimP,1);
    for (std::size_t i=0;i<dimP;++i)
      alpha(i) = 1;
    bool success = optim::sumt(alpha,objfn,nullptr,constrfn,nullptr);
    assert(success);

    for (std::size_t i=0;i<dimP;++i)
      ret[i] = alpha(i);
    return ret;
  };
}
