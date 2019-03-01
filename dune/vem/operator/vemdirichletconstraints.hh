#ifndef DUNE_VEM_DIRICHLETCONSTRAINTS_HH
#define DUNE_VEM_DIRICHLETCONSTRAINTS_HH

#include <dune/fem/function/common/scalarproducts.hh>
#include <dune/fem/schemes/dirichletconstraints.hh>
#include <dune/vem/space/interpolation.hh>

namespace Dune {

  template < class Model, class DiscreteFunctionSpace >
  class VemDirichletConstraints : public DirichletConstraints<Model,DiscreteFunctionSpace>
  {
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
    typedef FieldVector<int, localBlockSize> DirichletBlock;
    typedef FieldVector<int, localBlockSize> ModelDirichletBlock;

    VemDirichletConstraints( ModelType &model, const DiscreteFunctionSpaceType& space )
      : BaseType(model,space) {}

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
      if (op == Operation::set)
        this->operator()(u,w);

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
      const EntityType &entity = localMatrix.rangeEntity();
      const auto &slaveDofs = localMatrix.rangeSpace().slaveDofs();

      // get number of basis functions
      const int localBlocks = space_.blockMapper().numDofs( entity );

      // map local to global dofs
      std::vector<std::size_t> globalBlockDofs(localBlocks);
      // obtain all DofBlocks for this element
      space_.blockMapper().map( entity, globalBlockDofs );
      std::vector<bool> mask( localBlocks*localBlockSize );
      Dune::Vem::agglomerationVEMInterpolation<space_.polynomialOrder,space_.conforming>( space_.blockMapper().indexSet() )( entity, mask );

      // counter for all local dofs (i.e. localBlockDof * localBlockSize + ... )
      int localDof = 0;
      // iterate over face dofs and set unit row
      for( int localBlockDof = 0 ; localBlockDof < localBlocks; ++ localBlockDof )
      {
        int global = globalBlockDofs[localBlockDof];
        for( int l = 0; l < localBlockSize; ++ l, ++ localDof )
        {
          if( dirichletBlocks_[global][l] && mask[localDof] )
          {
            // clear all other columns
            localMatrix.clearRow( localDof );

            // set diagonal to 1
            double value = slaveDofs.isSlave( global )? 0.0 : 1.0;
            localMatrix.set( localDof, localDof, value );
          }
        }
      }
    }
    //! set the dirichlet points to exact values
    template< class LocalFunctionType >
    void dirichletDofTreatment( LocalFunctionType &&wLocal ) const
    {
      // get entity
      const typename LocalFunctionType::EntityType &entity = wLocal.entity();
      model_.init(entity);

      // get number of Lagrange Points
      const int localBlocks = space_.blockMapper().numDofs( entity );

      // map local to global BlockDofs
      std::vector< std::size_t > globalBlockDofs( localBlocks );
      space_.blockMapper().map( entity, globalBlockDofs );
      std::vector< double > valuesModel( localBlocks*localBlockSize );
      std::vector< bool > mask( localBlocks*localBlockSize );
      Dune::Vem::agglomerationVEMInterpolation<space_.polynomialOrder,space_.conforming>( space_.blockMapper().indexSet() ) ( entity, mask );

      int localDof = 0;

      for( int localBlock = 0; localBlock < localBlocks; ++localBlock )
      {
        // store result to dof vector
        int global = globalBlockDofs[ localBlock ];
        for( int l = 0; l < localBlockSize; ++l, ++localDof )
          if( dirichletBlocks_[ global ][ l ] && mask[ localDof ])
          {
            std::fill(valuesModel.begin(),valuesModel.end(),0);
            Dune::Vem::agglomerationVEMInterpolation<space_.polynomialOrder,space_.conforming>( space_.blockMapper().indexSet() )
            ( entity, typename BaseType::BoundaryWrapper(model_,dirichletBlocks_[global][l]),
                valuesModel );
            // store result
            assert( (unsigned int)localDof < wLocal.size() );
            wLocal[ localDof ] = valuesModel[ localDof ];
          }
      }
    }
    template< class LocalFunctionType >
    void dirichletDofTreatment( const LocalFunctionType &uLocal,
                                LocalFunctionType &&wLocal, Operation op ) const
    {
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
      std::vector< bool > mask( localBlocks*localBlockSize );
      assert( uLocal.size() == values.size() );
      for (unsigned int i=0;i<uLocal.size();++i)
        values[i] = uLocal[i];
      Dune::Vem::agglomerationVEMInterpolation<space_.polynomialOrder,space_.conforming>( space_.blockMapper().indexSet() )( entity, mask );

      int localDof = 0;

      for( int localBlock = 0; localBlock < localBlocks; ++localBlock )
      {
        // store result to dof vector
        int global = globalBlockDofs[ localBlock ];
        for( int l = 0; l < localBlockSize; ++l, ++localDof )
        {
          if( dirichletBlocks_[ global ][l] && mask[ localDof ])
          {
            if (op == Operation::sub)
            {
              std::fill(valuesModel.begin(),valuesModel.end(),0);
              Dune::Vem::agglomerationVEMInterpolation<space_.polynomialOrder,space_.conforming>( space_.blockMapper().indexSet() )
              ( entity, typename BaseType::BoundaryWrapper(model_,dirichletBlocks_[global][l]),
                  valuesModel );
              values[ localDof ] -= valuesModel[ localDof ];
            }
            assert( (unsigned int)localDof < wLocal.size() );
            wLocal[ localDof ] = values[ localDof ];
          }
        }
      }
    }
  private:
    using BaseType::model_;
    using BaseType::space_;
    using BaseType::dirichletBlocks_;
  };

} // end namespace Dune
#endif
