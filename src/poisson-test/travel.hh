/*
 * traverse.hh
 *
 *  Created on: 16 Sep 2015
 *      Author: gcd3
 */

#ifndef SRC_TRAVERSE_HH_
#define SRC_TRAVERSE_HH_


#include <dune/grid/common/scsgmapper.hh>
#include <dune/grid/common/mcmgmapper.hh>
#include<dune/grid/common/universalmapper.hh>



template <int commCodim>
struct LayoutWrapper
{
  template <int dim>
  struct Layout
  {
    bool contains(Dune::GeometryType gt)
    { return gt.dim() == dim - commCodim;  }
  };
};


template<class GridType>
void traverse(const GridType& grid){

  const int dim=GridType::dimension;
  typedef typename GridType::LeafGridView GridView;
  const GridView &gv = grid.leafGridView();
  typedef typename GridView::template Codim<0>::Iterator EIterator;
  typedef typename GridView::template Codim<1>::Iterator EdgeIterator;
  typedef typename GridView::template Codim<2>::Iterator VtxIterator;
  typedef typename GridView::IntersectionIterator IntersectionIterator;

  // ************************************************************************



  const EIterator &eend = gv.template end<0>();
  for(EIterator eit = gv.template begin<0>(); eit != eend; ++eit)
  {
    //    std::cout << "e.cntr() =  " << eit->geometry().center() << std::endl;

    if (eit->partitionType() == Dune::InteriorEntity) {
      std::cout << "InteriorEntity: e.cntr() =  " << eit->geometry().center() << std::endl;
    }

    if (eit->partitionType() == Dune::BorderEntity) {
      std::cout << "BorderEntity: e.cntr() =  " << eit->geometry().center() << std::endl;
    }

    if (eit->partitionType() == Dune::OverlapEntity) {
      std::cout << "OverlapEntity: e.cntr() =  " << eit->geometry().center() << std::endl;
    }

    if (eit->partitionType() == Dune::FrontEntity) {
      std::cout << "FrontEntity: e.cntr() =  " << eit->geometry().center() << std::endl;
    }

    if (eit->partitionType() == Dune::GhostEntity) {
      std::cout << "GhostEntity: e.cntr() =  " << eit->geometry().center() << std::endl;
    }



    //      const IntersectionIterator isend = gv.iend(*eit);
    //      for( IntersectionIterator is = gv.ibegin(*eit); is != isend; ++is )
    //      {
    //        std::cout << "chk" << std::endl;
    //      }





  }
  // ************************************************************************



  const Dune::PartitionIteratorType pit_all=Dune::All_Partition  ;
  typedef typename GridType::template Codim<0>::template Partition<pit_all>::LeafIterator AllElementIterator;

  typedef typename GridView::IntersectionIterator IntersectionIterator;

  int all_ElmCount = 0;
  for (AllElementIterator it = grid.template leafbegin<0,pit_all>(); it!=grid.template leafend<0,pit_all>(); ++it) {
    if ( it->partitionType() == Dune::GhostEntity || it->partitionType() == Dune::BorderEntity)
      continue;
    IntersectionIterator isIt = gv.ibegin(*it);
    const IntersectionIterator &isEndIt = gv.iend(*it);
    int numedges_on_domain_boundary = 0;
    std::cout << "coe: " << it->geometry().center() << std::endl;
    for (; isIt != isEndIt; ++isIt) {
      if(isIt->boundary()) ++ numedges_on_domain_boundary;
    }
    std::cout << "numedges_on_domain_boundary = " << numedges_on_domain_boundary << std::endl;
    ++all_ElmCount;
  }




  //-------------------------------------------------------------
  // test-parallel-ug.cc
  typedef Dune::MultipleCodimMultipleGeomTypeMapper<GridView, LayoutWrapper<0>::template Layout>
  MapperType;
  MapperType mapper(gv);



  typedef std::vector<Dune::FieldVector<double,1> > mydatatype;
  mydatatype entityIndex(gv.size(0), -1e10);

  for (AllElementIterator it = grid.template leafbegin<0,pit_all>(); it!=grid.template leafend<0,pit_all>(); ++it) { // element loop 2001
    int numberOfSubEntities = it->template count<1>();
    std::cout << "no of subentities = " << numberOfSubEntities << std::endl;


    for (int k = 0; k < numberOfSubEntities; k++)
    {
      typedef typename GridView::template Codim<0>::Entity Element;
      typedef typename Element::template Codim<0>::EntityPointer EntityPointer;
      //      const EntityPointer entityPointer(it->template subEntity<1>(k));
      //      entityIndex[mapper.map(*entityPointer)]   = gmapper.map(*entityPointer);
      //            std::cout << ">> " << gmapper.map(*entityPointer) << std::endl;


    }




  } //element loop 2001


  //-------------------------------------------------------------
  // from mapper-example.cc
  typedef typename GridType::template Codim<dim>::template Partition<pit_all>::LeafIterator AllVertexIterator;
  for (AllVertexIterator it = grid.template leafbegin<dim,pit_all>(); it!=grid.template leafend<dim,pit_all>(); ++it) { // element loop 2001
    // If the vertex is an interior vertex, it is owned by this process
    if (it->partitionType()==Dune::BorderEntity){

      std::cout << "you have border vtx" << it->geometry().center()  << std::endl;

    }
  }






  const Dune::PartitionIteratorType pit_ib=Dune::Interior_Partition  ;
  typedef typename GridType::template Codim<0>::template Partition<pit_ib>::LeafIterator IBElemIterator;
  int ibelmcount = 0;
  int num_vertices_on_polygon_boundary = 0;
  double area_of_polygon = 0.0;
  int maxPolygonVertex = 100;
  typedef Dune::DynamicVector<int> VectorofInts;

  int j = 0;
  for (IBElemIterator it = grid.template leafbegin<0,pit_ib>(); it!=grid.template leafend<0,pit_ib>(); ++it) { // element loop 2001
    ++ibelmcount;

    Dune::GeometryType gt = it->type();
    typedef typename IBElemIterator::Entity::Geometry LeafGeometry;
    const LeafGeometry geo = it->geometry();

    auto& ref = Dune::ReferenceElements<double,dim>::general(gt);

    typedef typename GridView::IndexSet LeafIndexSet;
    const LeafIndexSet& set = gv.indexSet();
    area_of_polygon = area_of_polygon + geo.volume();


    IntersectionIterator isIt = gv.ibegin(*it);
    const IntersectionIterator &isEndIt = gv.iend(*it);

    std::cout << "ur elm cntr: " << it->geometry().center() << std::endl;
    int somecount = 0;
    int othercount = 0;

    for (; isIt != isEndIt; ++isIt) {
      if ((isIt->boundary()) || ( isIt->neighbor() && isIt->inside().partitionType() == Dune::InteriorEntity && isIt->outside().partitionType() == Dune::GhostEntity))
        //      if ((!isIt->boundary()) || ( isIt->neighbor() && isIt->outside().partitionType() == Dune::InteriorEntity))
        //      {
        //        somecount++;
        //      }
        //      else
        //      {
      {
        for (int i = 0; i < ref.size(isIt->indexInInside(),1,dim); ++i)
        {

          // get the vertex index now:*****

          int local_vertex_index = ref.subEntity(isIt->indexInInside(),1,i,dim);



          //

          //

          std::cout << "edge centre" << it->geometry().global( ref.position(i,1) ) << std::endl;
          int indexi = set.subIndex(*it,
              ref.subEntity(isIt->indexInInside(), 1, i, dim), dim);
          std::cout << "polygon vertex = " << local_vertex_index << " " << geo.corner(local_vertex_index) << std::endl;


          ++j;


          // get the vertex index now:*****

          //          std::cout << "see: " <<  isIt->geometry().center() << std::endl;
          std::cout << "i: " << indexi << std::endl;


        }
      }
      ++othercount;
      //      }

    }
    num_vertices_on_polygon_boundary = num_vertices_on_polygon_boundary + othercount;
    //    std::cout << "criterion count = " << somecount++ << std::endl;
    //    std::cout << "no. vtcs on polygon border = " << othercount++ << std::endl;

  }










  std::cout << " -----------------------------------------------------------"<< std::endl;

  std::cout << "total ib elms: " << ibelmcount << std::endl;
  std::cout << "total polygon vrtcs " << num_vertices_on_polygon_boundary << std::endl;
  std::cout << "total area_of_polygon = " << area_of_polygon << std::endl;
  std::cout << " -----------------------------------------------------------"<< std::endl;







  std::cout << "InteriorBorder_Partition_elms =  " << all_ElmCount << std::endl;





}  // void traverse()

#endif /* SRC_TRAVERSE_HH_ */
