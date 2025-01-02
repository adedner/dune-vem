#ifndef DUNE_VEM_DIRICHLETCONSTRAINTS_HH
#define DUNE_VEM_DIRICHLETCONSTRAINTS_HH

#include <dune/common/exceptions.hh>
#include <dune/fem/function/common/scalarproducts.hh>
#include <dune/fem/schemes/dirichletconstraints.hh>
#include <dune/fem/operator/common/temporarylocalmatrix.hh>
#include <dune/fem/function/localfunction/const.hh>
#include <dune/fem/function/localfunction/mutable.hh>
#include <dune/fem/common/bindguard.hh>

// #include <dune/vem/space/interpolation.hh>
#include <dune/vem/misc/vector.hh>

namespace Dune {

  template < class Model, class DiscreteFunctionSpace, bool useIdentity,
             int bndMask > // mask | 1: fix value
                           // mask | 2: fix normal derivative
                           // -> 2nd order: use mask=1
                           //    4th order: use mask=3 for fixing both
  class VemDirichletConstraints
  : public DirichletConstraints<Model,DiscreteFunctionSpace, useIdentity>
  {
    static_assert( 1<=bndMask && bndMask<=3 );
    typedef DirichletConstraints<Model,DiscreteFunctionSpace, useIdentity> BaseType;
  public:
    enum Operation { set = 0, sub = 1, add = 2 };
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
      { impl_.dDirichlet(bndId_,Dune::Fem::coordinate(x),ret); }
    };
    VemDirichletConstraints( ModelType &model, const DiscreteFunctionSpaceType& space )
      : BaseType(model,space)
    {}

    bool applyConstraint(char maskValue) const
    {
      // maskValue = 0: not on bnd
      //           = 1: a value dof on bnd
      //           = 2: a derivative dof on bnd
      if (maskValue>3) {std::cout << "applyConstraint got wrong mask value: " << maskValue << std::endl; assert(false);}
      if (maskValue<=0) return false;
      return ! ((bndMask & maskValue) == 0);
    }

    template < class DiscreteFunctionType >
    void operator ()( const DiscreteFunctionType& u, DiscreteFunctionType& w ) const
    {
      BaseType::operator()(u,w);
    }

    template < class DiscreteFunctionType >
    void operator ()( const typename DiscreteFunctionType::RangeType& value, DiscreteFunctionType& w ) const
    {
      // not yet correctly implemented
      assert(false);
      BaseType::updateDirichletDofs();
      if( BaseType::hasDirichletDofs_ )
      {
        Dune::Fem::MutableLocalFunction< DiscreteFunctionType > wLocal( w );
        for( const EntityType &entity : space_ )
        {
          auto wGuard = Dune::Fem::bindGuard( wLocal, entity );
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
              if( dirichletBlocks_[ global ][ l ] && applyConstraint(mask[ localBlock ]))
                wLocal[ localDof ] = 0;
            }
          }
        }
      }
    }

    template < class DiscreteFunctionType >
    void operator ()( DiscreteFunctionType& w ) const
    {
      BaseType::updateDirichletDofs();
      if( BaseType::hasDirichletDofs_ )
      {
        Dune::Fem::MutableLocalFunction< DiscreteFunctionType > wLocal( w );
        for( const EntityType &entity : space_ )
        {
          auto wGuard = Dune::Fem::bindGuard( wLocal, entity );
          dirichletDofTreatment( wLocal );
        }
      }
    }
    template < class DiscreteFunctionType >
    void operator ()( const DiscreteFunctionType &u,
                      DiscreteFunctionType& w, Operation op) const
    {
      BaseType::updateDirichletDofs();
      if( BaseType::hasDirichletDofs_ )
      {
        Dune::Fem::ConstLocalFunction< DiscreteFunctionType > uLocal( u );
        Dune::Fem::MutableLocalFunction< DiscreteFunctionType > wLocal( w );
        for( const auto &entity : space_ )
        {
          auto uGuard = Dune::Fem::bindGuard( uLocal, entity );
          auto wGuard = Dune::Fem::bindGuard( wLocal, entity );
          dirichletDofTreatment( uLocal, wLocal, op );
        }
      }
    }

    template < class GridFunctionType, class DiscreteFunctionType,
             typename = std::enable_if_t< std::is_base_of<Dune::Fem::HasLocalFunction, GridFunctionType>::value > >
    void operator ()( const GridFunctionType &u,
                      DiscreteFunctionType& w, Operation op=Operation::setDF ) const
    {
      DUNE_THROW( Dune::NotImplemented, "constraints can currently only be set given a discrete function not a general grid function");
    }

    template <class LinearOperator>
    void applyToOperator( LinearOperator& linearOperator ) const
    {
      typedef typename DiscreteFunctionSpaceType :: IteratorType IteratorType;
      typedef typename IteratorType :: Entity EntityType;
      BaseType::updateDirichletDofs();

      if( BaseType::hasDirichletDofs_ )
      {
        typedef typename DiscreteFunctionSpaceType :: BlockMapperType BlockMapperType;
        Dune::Fem::NonBlockMapper< BlockMapperType, localBlockSize > mapper( space_.blockMapper() );

        std::vector<std::size_t> globalBlockDofs;
        std::vector<std::size_t> globalDofs;

        // storage for unit rows and auxiliary rows, should be unique and sorted
        std::set<std::size_t> unitRows;
        std::set<std::size_t> auxRows;

        const auto& space = space_; // linearOperator.rangeSpace();

        const IteratorType end = space_.end();
        for( IteratorType it = space_.begin(); it != end; ++it )
        {
          const EntityType &entity = *it;

          // get number of basis functions
          const int localBlocks = space.blockMapper().numDofs( entity );

          // map local to global dofs
          globalBlockDofs.resize(localBlocks);
          // obtain all DofBlocks for this element
          space.blockMapper().map( entity, globalBlockDofs );

          // obtain all non-blocked dofs
          globalDofs.resize(localBlocks * localBlockSize);
          mapper.map( entity, globalDofs );

          Vem::Std::vector< char > mask( localBlocks );
          space_.interpolation()( entity, mask );

          // counter for all local dofs (i.e. localBlockDof * localBlockSize + ... )
          int localDof = 0;
          // iterate over face dofs and set unit row
          for( int localBlockDof = 0 ; localBlockDof < localBlocks; ++ localBlockDof )
          {
            int global = globalBlockDofs[localBlockDof];
            // all diagonals were set to one anyway independent of auxDof or not
            // so don't need this distinction anymore
            // std::set<std::size_t>& rows = auxiliaryDofs.contains( global ) ? auxRows : unitRows;
            std::set<std::size_t>& rows = useIdentity ?  unitRows : auxRows;
            for( int l = 0; l < localBlockSize; ++ l, ++ localDof )
            {
              if( dirichletBlocks_[global][l] &&
                  applyConstraint(mask[localBlockDof]) )
              {
                // push non-blocked dof
                rows.insert( globalDofs[ localDof ] );
              }
            }
          }
        } // end for elements

        // set unit and auxiliary rows at once
        linearOperator.setUnitRows( unitRows, auxRows );
      }
    }

  protected:
    //! set the dirichlet points to exact values
    template< class LocalFunctionType >
    void dirichletDofTreatment( LocalFunctionType &wLocal ) const
    {
      // assert(false);
      // get entity
      const auto &entity = wLocal.entity();
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
            space_.interpolation()( entity, BoundaryWrapper(model_,dirichletBlocks_[global][l]),  valuesModel );
            // store result
            assert( (unsigned int)localDof < wLocal.size() );
            wLocal[ localDof ] = valuesModel[ localDof ];
          }
        }
      }
    }
    template< class LocalFunctionType, class LocalContributionType >
    void dirichletDofTreatment( const LocalFunctionType &uLocal,
                                LocalContributionType &wLocal, Operation op ) const
    {
      // get entity
      const auto &entity = wLocal.entity();
      if (op == Operation::sub || op == Operation::add)
        model_.init(entity);

      // get number of Lagrange Points
      const int localBlocks = space_.blockMapper().numDofs( entity );

      // map local to global BlockDofs
      std::vector< std::size_t > globalBlockDofs( localBlocks );
      space_.blockMapper().map( entity, globalBlockDofs );

      std::vector<double> values( localBlocks*localBlockSize );
      std::vector<double> valuesModel( localBlocks*localBlockSize );

      assert( uLocal.size() == values.size() );
      assert( wLocal.size() == values.size() );

      Vem::Std::vector< char > mask( localBlocks );
      space_.interpolation()( entity, mask );

      if constexpr ( useIdentity )
      {
        assert( LocalFunctionType::FunctionSpaceType::dimRange ==
                DiscreteFunctionSpaceType::FunctionSpaceType::dimRange );
        for (unsigned int i=0;i<uLocal.size();++i)
          values[i] = uLocal[i];
      }

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
              space_.interpolation() ( entity, BoundaryWrapper(model_,dirichletBlocks_[global][l]), valuesModel );
              //auto v = values[ localDof ];
              values[ localDof ] -= valuesModel[ localDof ];
              assert(std::abs(values[localDof]) < 1e-8);
            }
            else if (op == Operation::add)
            {
              std::fill(valuesModel.begin(),valuesModel.end(),0);
              space_.interpolation() ( entity, BoundaryWrapper(model_,dirichletBlocks_[global][l]), valuesModel );
              //auto v = values[ localDof ];
              values[ localDof ] += valuesModel[ localDof ];
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
