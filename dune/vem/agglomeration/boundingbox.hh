#ifndef DUNE_VEM_AGGLOMERATION_BOUNDINGBOX_HH
#define DUNE_VEM_AGGLOMERATION_BOUNDINGBOX_HH

#include <cassert>
#include <cstddef>

#include <array>
#include <utility>
#include <vector>

#include <dune/python/pybind11/pybind11.h>

#include <dune/common/fvector.hh>

#include <dune/vem/agglomeration/agglomeration.hh>

namespace Dune
{

  namespace Vem
  {

    // BoundingBox
    // -----------

    template< class GridPart >
    struct BoundingBox : public
      std::pair< FieldVector< typename GridPart::ctype, GridPart::dimensionworld >, FieldVector< typename GridPart::ctype, GridPart::dimensionworld > >
    {
      typedef FieldVector< typename GridPart::ctype, GridPart::dimensionworld > CoordinateType;
      typedef std::pair< CoordinateType, CoordinateType > BaseType;
      typedef std::tuple<CoordinateType,CoordinateType,
                         CoordinateType,CoordinateType, double> ReturnType;
      using BaseType::BaseType;
      double &volume() { return volume_; }
      const double &volume() const { return volume_; }
      const CoordinateType &lower() const { return std::get<0>(rotation_); }
      const CoordinateType &upper() const { return std::get<1>(rotation_); }
      const CoordinateType &xAxis() const { return std::get<2>(rotation_); }
      const CoordinateType &yAxis() const { return std::get<3>(rotation_); }
      CoordinateType transform(CoordinateType x) const
      {
        // std::cout << lower() << " , " << upper() << "    "
        //          << xAxis() << " , " << yAxis() << "    " << x << " -> " << std::flush;
        x -= lower();
        CoordinateType y;
        transform_.mtv(x,y);
        // std::cout << y << std::endl;
        assert(-1e-8 < y[0] && y[0] < 1+1e-8);
        assert(-1e-8 < y[1] && y[1] < 1+1e-8);
        return std::move(y);
      }
      void gradientTransform(CoordinateType &g, bool transpose) const
      {
        CoordinateType tmp(g);
        if (transpose)
          transform_.mtv(tmp,g);
        else
          transform_.mv(tmp,g);
      }
      const double &diameter() const { return std::get<4>(rotation_); }

      void set(ReturnType rotation)
      {
        rotation_ = std::move(rotation);
        transform_[0] = xAxis();
        transform_[1] = yAxis();
        transform_.invert();
      }
      void set(pybind11::object obj) { set( obj.cast<ReturnType>() ); }
      private:
      ReturnType rotation_;
      double volume_ = 0;
      Dune::FieldMatrix< typename GridPart::ctype, GridPart::dimensionworld, GridPart::dimensionworld > transform_;
    };

    // agglomerateBoundingBoxes
    // ------------------------

    template< class GridPart >
    inline static std::vector< BoundingBox< GridPart > > boundingBoxes ( const Agglomeration< GridPart > &agglomeration )
    {
      typedef typename GridPart::template Codim< 0 >::GeometryType GeometryType;

      BoundingBox< GridPart > emptyBox;
      for( int k = 0; k < GridPart::dimensionworld; ++k )
      {
        emptyBox.first[ k ] = std::numeric_limits< typename GridPart::ctype >::max();
        emptyBox.second[ k ] = std::numeric_limits< typename GridPart::ctype >::min();
      }

      std::vector< BoundingBox< GridPart > > boundingBoxes( agglomeration.size(), emptyBox );
      std::vector<std::vector<std::vector<double>>> polygonPoints( agglomeration.size() );
      for( const auto element : elements( static_cast< typename GridPart::GridViewType >( agglomeration.gridPart() ), Partitions::interiorBorder ) )
      {
        BoundingBox< GridPart > &bbox = boundingBoxes[ agglomeration.index( element ) ];
        std::vector<std::vector<double>> &points = polygonPoints[ agglomeration.index( element) ];
        bbox.volume() += element.geometry().volume();
        const GeometryType geometry = element.geometry();
        for( int i = 0; i < geometry.corners(); ++i )
        {
          const typename GeometryType::GlobalCoordinate corner = geometry.corner( i );
          for( int k = 0; k < GridPart::dimensionworld; ++k )
          {
            bbox.first[ k ] = std::min( bbox.first[ k ], corner[ k ] );
            bbox.second[ k ] = std::max( bbox.second[ k ], corner[ k ] );
          }
          points.push_back({corner[0],corner[1]});
        }

      }
      for (int i=0;i<boundingBoxes.size();++i)
      {
        BoundingBox< GridPart > &bbox = boundingBoxes[ i ];
        std::vector<std::vector<double>> &points = polygonPoints[ i ];
        auto pyBBox   = pybind11::module::import("dune.vem.bbox");
        auto bboxobj  = pyBBox.attr("rotatedBBox")(points);
        bbox.set( bboxobj );
        // std::cout << bbox.lower() << "   " << bbox.upper() << "     ";
        // std::cout << bbox.xAxis() << "   " << bbox.yAxis() << "     ";
        // std::cout << bbox.diameter() << std::endl;
      }
      return std::move( boundingBoxes );
    }

  } // namespace Vem

} // namespace Dune

#endif // #define DUNE_VEM_AGGLOMERATION_BOUNDINGBOX_HH
