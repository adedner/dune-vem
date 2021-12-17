#ifndef DUNE_VEM_DIRICHLETCONSTRAINTS_HH
#define DUNE_VEM_DIRICHLETCONSTRAINTS_HH

#include <dune/fem/function/common/scalarproducts.hh>
#include <dune/fem/schemes/dirichletconstraints.hh>
// #include <dune/vem/space/interpolation.hh>
#include <dune/vem/misc/vector.hh>

namespace Dune {

  template < class Model, class DiscreteFunctionSpace,
             int bndMask > // mask | 1: fix value
                           // mask | 2: fix normal derivative
                           // -> 2nd order: use mask=1
                           //    4th order: use mask=3 for fixing both
  class VemDirichletConstraints : public DirichletConstraints<Model,DiscreteFunctionSpace>
  {
    static_assert( 1<=bndMask && bndMask<=3 );
    typedef DirichletConstraints<Model,DiscreteFunctionSpace> BaseType;
  public:
    enum Operation { set = 0, sub = 1 };
    typedef Model ModelType;
    typedef DiscreteFunctionSpace DiscreteFunctionSpaceType;
    typedef typename DiscreteFunctionSpaceType::RangeType RangeType;
    typedef typename DiscreteFunctionSpaceType::JacobianRangeType JacobianRangeType;
    typedef typename DiscreteFunctionSpaceType::EntityType EntityType;

    //! type of grid partition
    typedef typename DiscreteFunctionSpaceType :: GridPartType GridPartType;
    //! type of grid
    typedef typename DiscreteFunctionSpaceType :: GridType GridType;

    static const int localBlockSize = DiscreteFunctionSpaceType :: localBlockSize ;

    class BoundaryWrapper
    {
      const ModelType& impl_;
      int bndId_;
      public:
      typedef typename DiscreteFunctionSpaceType::EntityType EntityType;
      typedef typename DiscreteFunctionSpaceType::FunctionSpaceType FunctionSpaceType;
      typedef typename DiscreteFunctionSpaceType::DomainType DomainType;
      typedef typename DiscreteFunctionSpace::RangeType RangeType;
      typedef typename DiscreteFunctionSpace::JacobianRangeType JacobianRangeType;
      typedef typename DiscreteFunctionSpace::HessianRangeType HessianRangeType;
      static const int dimRange = RangeType::dimension;
      BoundaryWrapper( const ModelType& impl, int bndId )
      : impl_( impl ), bndId_(bndId) {}
      template <class Point>
      void evaluate( const Point& x, RangeType& ret ) const
      { impl_.dirichlet(bndId_,Dune::Fem::coordinate(x),ret); }
      template <class Point>
      void jacobian( const Point& x, JacobianRangeType& ret ) const
      { ret = JacobianRangeType(0); }
    };
    VemDirichletConstraints( ModelType &model, const DiscreteFunctionSpaceType& space )
      : BaseType(model,space)
    {}

    bool applyConstraint(char maskValue) const
    {
      // maskValue = 0: not on bnd
      //           = 1: a value dof on bnd
      //           = 2: a derivative dof on bnd
      if (maskValue>2) {std::cout << "applyConstraint got wrong mask value: " << maskValue << std::endl; assert(false);}
      switch (bndMask)
      {
        case 1: return (maskValue == 1);
        case 2: return (maskValue == 2);
        case 3: return (maskValue >= 1);
      }
      return false; // can't be reached
    }

#if 0
    template < class DiscreteFunctionType >
    void operator ()( const DiscreteFunctionType& u, DiscreteFunctionType& w ) const
    {
      BaseType::operator()(u,w);
    }
    template < class DiscreteFunctionType >
    void operator ()( const typename DiscreteFunctionType::RangeType& value, DiscreteFunctionType& w ) const
    {
      BaseType::operator()(value,w);
    }
#endif

    template < class DiscreteFunctionType >
    void operator ()( const typename DiscreteFunctionType::RangeType& value, DiscreteFunctionType& w ) const
    {
      return;
      this->operator()(w);
    }
    template < class DiscreteFunctionType >
    void operator ()( DiscreteFunctionType& w ) const
    {
      BaseType::updateDirichletDofs();
      if( BaseType::hasDirichletDofs_ )
        for( const EntityType &entity : space_ )
          dirichletDofTreatment( w.localFunction( entity ) );

    }
    template < class DiscreteFunctionType >
    void operator ()( const DiscreteFunctionType &u,
                      DiscreteFunctionType& w, Operation op) const
    {
      BaseType::updateDirichletDofs();
      if( BaseType::hasDirichletDofs_ )
        for( const auto &entity : space_ )
          dirichletDofTreatment( u.localFunction(entity), w.localFunction(entity), op );
    }
    template <class LinearOperator>
    void applyToOperator( LinearOperator& linearOperator ) const
    {
      BaseType::updateDirichletDofs();
      if( BaseType::hasDirichletDofs_ )
        for( const auto &entity : space_ )
          dirichletDofsCorrectOnEntity( linearOperator.localMatrix(entity,entity) );

    }

  protected:
    template< class LocalLinearOperator >
    void dirichletDofsCorrectOnEntity ( LocalLinearOperator &&localMatrix ) const
    {
      std::cout << "dirichletDofsCorrectOnEntity(localMatrix): ";
      const EntityType &entity = localMatrix.rangeEntity();
      const auto &slaveDofs = localMatrix.rangeSpace().slaveDofs();

      // get number of basis functions
      const int localBlocks = space_.blockMapper().numDofs( entity );
      // std::cout << "   " << localBlocks << " : ";

      // map local to global dofs
      std::vector<std::size_t> globalBlockDofs(localBlocks);
      // obtain all DofBlocks for this element
      space_.blockMapper().map( entity, globalBlockDofs );
      // std::cout << "global=";
      // for (auto g : globalBlockDofs)
      //   std::cout << g << ",";
      Vem::Std::vector< char > mask( localBlocks );
      space_.interpolation()( entity, mask );
      // std::cout << " mask=";
      // for (auto m : mask)
      //   std::cout << (m==1)?1:0 << ",";
      // std::cout << std::endl << "   ";
      // counter for all local dofs (i.e. localBlockDof * localBlockSize + ... )
      int localDof = 0;
      // iterate over face dofs and set unit row
      for( int localBlockDof = 0 ; localBlockDof < localBlocks; ++ localBlockDof )
      {
        int global = globalBlockDofs[localBlockDof];
        for( int l = 0; l < localBlockSize; ++ l, ++ localDof )
        {
          // std::cout << "(" << localBlockDof << "," << l << ") ";
          if( dirichletBlocks_[global][l] &&
              applyConstraint(mask[localBlockDof]) )
          {
            std::cout << global << " ";
            // clear all other columns
            localMatrix.clearRow( localDof );

            // set diagonal to 1
            double value = slaveDofs.isSlave( global )? 0.0 : 1.0;
            localMatrix.set( localDof, localDof, value );
          }
        }
      }
      std::cout << std::endl;
    }
    //! set the dirichlet points to exact values
    template< class LocalFunctionType >
    void dirichletDofTreatment( LocalFunctionType &&wLocal ) const
    {
      std::cout << "dirichletDofsCorrectOnEntity(wLocal): ";
      // get entity
      const typename LocalFunctionType::EntityType &entity = wLocal.entity();
      model_.init(entity);

      // get number of Lagrange Points
      const int localBlocks = space_.blockMapper().numDofs( entity );

      // map local to global BlockDofs
      std::vector< std::size_t > globalBlockDofs( localBlocks );
      space_.blockMapper().map( entity, globalBlockDofs );
      std::vector< double > valuesModel( localBlocks*localBlockSize );
      Vem::Std::vector< char > mask( localBlocks );
      space_.interpolation()( entity, mask );

      int localDof = 0;

      for( int localBlock = 0; localBlock < localBlocks; ++localBlock )
      {
        // store result to dof vector
        int global = globalBlockDofs[ localBlock ];
        for( int l = 0; l < localBlockSize; ++l, ++localDof )
        {
          if( dirichletBlocks_[ global ][ l ] &&
              applyConstraint(mask[ localBlock ]))
          {
            std::fill(valuesModel.begin(),valuesModel.end(),0);
            // space_.interpolation()( entity, BoundaryWrapper(model_,dirichletBlocks_[global][l]),  valuesModel );
            // store result
            assert( (unsigned int)localDof < wLocal.size() );
            wLocal[ localDof ] = valuesModel[ localDof ];
          }
        }
      }
      std::cout << std::endl;
    }
    template< class LocalFunctionType >
    void dirichletDofTreatment( const LocalFunctionType &uLocal,
                                LocalFunctionType &&wLocal, Operation op ) const
    {
      std::cout << "dirichletDofsCorrectOnEntity(uLocal,wLocal,op): ";
      // get entity
      const typename LocalFunctionType::EntityType &entity = wLocal.entity();
      if (op == Operation::sub)
        model_.init(entity);

      // get number of Lagrange Points
      const int localBlocks = space_.blockMapper().numDofs( entity );

      // map local to global BlockDofs
      std::vector< std::size_t > globalBlockDofs( localBlocks );
      space_.blockMapper().map( entity, globalBlockDofs );

      std::vector<double> values( localBlocks*localBlockSize );
      std::vector<double> valuesModel( localBlocks*localBlockSize );
      Vem::Std::vector< char > mask( localBlocks );
      assert( uLocal.size() == values.size() );
      assert( wLocal.size() == values.size() );
      for (unsigned int i=0;i<uLocal.size();++i)
        values[i] = uLocal[i];
      space_.interpolation()( entity, mask );

      int localDof = 0;

      for( int localBlock = 0; localBlock < localBlocks; ++localBlock )
      {
        // store result to dof vector
        int global = globalBlockDofs[ localBlock ];
        for( int l = 0; l < localBlockSize; ++l, ++localDof )
        {
          if( dirichletBlocks_[ global ][l] &&
              applyConstraint(mask[ localBlock ]) )
          {
            if (op == Operation::sub)
            {
              std::fill(valuesModel.begin(),valuesModel.end(),0);
              // space_.interpolation() ( entity, BoundaryWrapper(model_,dirichletBlocks_[global][l]), valuesModel );
              values[ localDof ] -= valuesModel[ localDof ];
            }
            assert( (unsigned int)localDof < wLocal.size() );
            wLocal[ localDof ] = values[ localDof ];
          }
        }
      }
      // std::cout << std::endl;
    }
  private:
    using BaseType::model_;
    using BaseType::space_;
    using BaseType::dirichletBlocks_;
  };

} // end namespace Dune
#endif
