#ifndef DUNE_DIRICHLETCONSTRAINTS_HH
#define DUNE_DIRICHLETCONSTRAINTS_HH

#include <dune/fem/function/common/scalarproducts.hh>

namespace Dune {

  template < class Model, class DiscreteFunctionSpace >
    class DirichletConstraints
    {
      public:
        typedef Model ModelType;
        typedef DiscreteFunctionSpace DiscreteFunctionSpaceType;

        //! type of grid partition
        typedef typename DiscreteFunctionSpaceType :: GridPartType GridPartType;
        //! type of grid
        typedef typename DiscreteFunctionSpaceType :: GridType GridType;

        // types for boundary treatment
        // ----------------------------
        typedef typename DiscreteFunctionSpaceType :: BlockMapperType BlockMapperType;

        DirichletConstraints( const ModelType &model, const DiscreteFunctionSpaceType& space )
          : model_(model),
          space_( space ),
          dirichletBlocks_(),
          // mark DoFs on the Dirichlet boundary
          hasDirichletDofs_( false ),
          sequence_( -1 )
      {
      }

        /*! treatment of Dirichlet-DoFs for given discrete function
         *
         *   \note A LagrangeDiscreteFunctionSpace is implicitly assumed.
         *
         *   \param[in]  u   discrete function providing the constraints
         *   \param[out] w   discrete function the constraints are applied to
         */
        template < class DiscreteFunctionType >
          void operator ()( const DiscreteFunctionType& u, DiscreteFunctionType& w ) const
          {
            updateDirichletDofs();

            // if Dirichlet Dofs have been found, treat them
            if( hasDirichletDofs_ )
            {
              typedef typename DiscreteFunctionType :: DofIteratorType DofIteratorType ;
              typedef typename DiscreteFunctionType :: ConstDofIteratorType ConstDofIteratorType ;

              ConstDofIteratorType uIt = u.dbegin();
              DofIteratorType wIt = w.dbegin();

              const unsigned int localBlockSize = DiscreteFunctionType :: DiscreteFunctionSpaceType ::
                localBlockSize ;
              // loop over all blocks
              const unsigned int blocks = u.space().blockMapper().size();
              for( unsigned int blockDof = 0; blockDof < blocks ; ++ blockDof )
              {
                if( dirichletBlocks_[ blockDof ] )
                {
                  // copy dofs of the block
                  for( unsigned int l = 0; l < localBlockSize ; ++ l, ++ wIt, ++ uIt )
                  {
                    assert( uIt != u.dend() );
                    assert( wIt != w.dend() );
                    (*wIt) = (*uIt);
                  }
                }
                else
                {
                  // increase dof iterators anyway
                  for( unsigned int l = 0; l < localBlockSize ; ++ l, ++ wIt, ++ uIt )
                  {}
                }
              }
              // w.communicate();
            }
          }

        /*! treatment of Dirichlet-DoFs for given discrete function
         *
         *   \note A LagrangeDiscreteFunctionSpace is implicitly assumed.
         *
         *   \param[in]  u   discrete function providing the constraints
         *   \param[out] w   discrete function the constraints are applied to
         */
        template < class GridFunctionType, class DiscreteFunctionType >
          void operator ()( const GridFunctionType& u, DiscreteFunctionType& w ) const
          {
            apply( u, w );
          }

        /*! treatment of Dirichlet-DoFs for solution and right-hand-side
         *
         *   delete rows for dirichlet-DoFs, setting diagonal element to 1.
         *
         *   \note A LagrangeDiscreteFunctionSpace is implicitly assumed.
         *
         *   \param[out] linearOperator  linear operator to be adjusted
         */
        template <class LinearOperator>
          void applyToOperator( LinearOperator& linearOperator ) const
          {
            updateDirichletDofs();

            typedef typename DiscreteFunctionSpaceType :: IteratorType IteratorType;
            typedef typename IteratorType :: Entity EntityType;

            // if Dirichlet Dofs have been found, treat them
            if( hasDirichletDofs_ )
            {
              const IteratorType end = space_.end();
              for( IteratorType it = space_.begin(); it != end; ++it )
              {
                const EntityType &entity = *it;
                // adjust linear operator
                dirichletDofsCorrectOnEntity( linearOperator, entity );
              }
            }
          }

      protected:
        template < class GridFunctionType, class DiscreteFunctionType >
          void apply( const GridFunctionType& u, DiscreteFunctionType& w ) const
          {
            updateDirichletDofs();

            typedef typename DiscreteFunctionSpaceType :: IteratorType IteratorType;
            typedef typename IteratorType :: Entity EntityType;

            // if Dirichlet Dofs have been found, treat them
            if( hasDirichletDofs_ )
            {
              const IteratorType end = space_.end();
              for( IteratorType it = space_.begin(); it != end; ++it )
              {
                const EntityType &entity = *it;
                dirichletDofTreatment( entity, u, w );
              }
              // w.communicate();
            }
          }

        /*! treatment of Dirichlet-DoFs for one entity
         *
         *   delete rows for dirichlet-DoFs, setting diagonal element to 1.
         *
         *   \note A LagrangeDiscreteFunctionSpace is implicitly assumed.
         *
         *   \param[in]  entity  entity to perform Dirichlet treatment on
         */
        template< class LinearOperator, class EntityType >
          void dirichletDofsCorrectOnEntity ( LinearOperator& linearOperator,
              const EntityType &entity ) const
          {
            // get slave dof structure (for parallel runs)   /*@LST0S@*/
            const auto &slaveDofs = linearOperator.rangeSpace().slaveDofs();

            typedef typename DiscreteFunctionSpaceType :: LagrangePointSetType
              LagrangePointSetType;
            const LagrangePointSetType &lagrangePointSet = space_.lagrangePointSet( entity );

            typedef typename LinearOperator :: LocalMatrixType LocalMatrixType;

            // get local matrix from linear operator
            LocalMatrixType localMatrix = linearOperator.localMatrix( entity, entity );

            // get number of basis functions
            const int localBlocks = lagrangePointSet.size();
            const int localBlockSize = DiscreteFunctionSpaceType :: localBlockSize ;

            // map local to global dofs
            std::vector<std::size_t> globalBlockDofs(localBlocks);
            // obtain all DofBlocks for this element
            space_.blockMapper().map( entity, globalBlockDofs );

            // counter for all local dofs (i.e. localBlockDof * localBlockSize + ... )
            int localDof = 0;
            // iterate over face dofs and set unit row
            for( int localBlockDof = 0 ; localBlockDof < localBlocks; ++ localBlockDof )
            {
              int global = globalBlockDofs[localBlockDof];
              if( dirichletBlocks_[ global ] )
              {
                for( int l = 0; l < localBlockSize; ++ l, ++ localDof )
                {
                  // clear all other columns
                  localMatrix.clearRow( localDof );

                  // set diagonal to 1
                  double value = slaveDofs.isSlave( global )? 0.0 : 1.0;
                  localMatrix.set( localDof, localDof, value );
                }
              }
              else
              {
                // increase localDof anyway
                localDof += localBlockSize ;
              }
            }
          }

        //! set the dirichlet points to exact values
        template< class EntityType, class GridFunctionType, class DiscreteFunctionType >
          void dirichletDofTreatment( const EntityType &entity,
              const GridFunctionType& u,
              DiscreteFunctionType &w ) const
          {
            typedef typename DiscreteFunctionType :: DiscreteFunctionSpaceType
              DiscreteSpaceType;
            typedef typename GridFunctionType :: LocalFunctionType GridLocalFunctionType;
            typedef typename DiscreteFunctionType :: LocalFunctionType LocalFunctionType;

            typedef typename DiscreteSpaceType :: LagrangePointSetType
              LagrangePointSetType;

            // get local functions of result
            LocalFunctionType wLocal = w.localFunction( entity );

            // get local functions of argument
            GridLocalFunctionType uLocal = u.localFunction( entity );

            const LagrangePointSetType &lagrangePointSet = space_.lagrangePointSet( entity );

            // get number of Lagrange Points
            const int numBlocks = lagrangePointSet.size();

            int localDof = 0;
            const int localBlockSize = DiscreteSpaceType :: localBlockSize ;

            // map local to global BlockDofs
            std::vector<std::size_t> globalBlockDofs(numBlocks);
            space_.blockMapper().map(entity,globalBlockDofs);

            // iterate over face dofs and set unit row
            for( int localBlock = 0 ; localBlock < numBlocks; ++ localBlock )
            {
              if( dirichletBlocks_[ globalBlockDofs[ localBlock ] ] )
              {
                typedef typename DiscreteFunctionSpaceType :: RangeType RangeType;
                RangeType phi( 0 );

                // evaluate data
                uLocal.evaluate( lagrangePointSet[ localBlock ], phi );

                // store result to dof vector
                for( int l = 0; l < localBlockSize ; ++ l, ++localDof )
                {
                  // store result
                  wLocal[ localDof ] = phi[ l ];
                }
              }
              else
              {
                // increase localDofs by block size
                localDof += localBlockSize ;
              }
            }
          }

      protected:
        // detect all DoFs on the Dirichlet boundary
        void updateDirichletDofs() const
        {
          if( sequence_ != space_.sequence() )
          {
            // only start search if Dirichlet boundary is present
            if( ! model_.hasDirichletBoundary() )
            {
              hasDirichletDofs_ = false ;
              return ;
            }

            // resize flag vector with number of blocks and reset flags
            const int blocks = space_.blockMapper().size() ;
            dirichletBlocks_.resize( blocks );
            for( int i=0; i<blocks; ++i )
              dirichletBlocks_[ i ] = false ;

            typedef typename DiscreteFunctionSpaceType :: IteratorType IteratorType;
            typedef typename IteratorType :: Entity EntityType;

            bool hasDirichletBoundary = false;
            const IteratorType end = space_.end();
            for( IteratorType it = space_.begin(); it != end; ++it )
            {
              const EntityType &entity = *it;
              // if entity has boundary intersections
              if( entity.hasBoundaryIntersections() )
              {
                hasDirichletBoundary |= searchEntityDirichletDofs( entity, model_ );
              }
            }

            // update sequence number
            sequence_ = space_.sequence();
            if( space_.gridPart().comm().size() > 1 )
            {
              try
              {
                DirichletBuilder handle( *this, space_ , space_.blockMapper() );
                space_.gridPart().communicate
                  ( handle, GridPartType::indexSetInterfaceType, ForwardCommunication );
              }
              // catch possible exceptions here to have a clue where it happend
              catch( const Exception &e )
              {
                std::cerr << e << std::endl;
                std::cerr << "Exception thrown in: " << __FILE__ << " line:" << __LINE__ << std::endl;
                abort();
              }
              hasDirichletDofs_ = space_.gridPart().grid().comm().max( hasDirichletBoundary );
            }
            else
            {
              hasDirichletDofs_ = hasDirichletBoundary;
            }
          }
        }

        // detect all DoFs on the Dirichlet boundary of the given entity
        template< class EntityType >
          bool searchEntityDirichletDofs( const EntityType &entity, const ModelType& model ) const
          {

            typedef typename DiscreteFunctionSpaceType :: LagrangePointSetType
              LagrangePointSetType;

            typedef typename DiscreteFunctionSpaceType :: GridPartType GridPartType;

            const int faceCodim = 1;
            typedef typename GridPartType :: IntersectionIteratorType
              IntersectionIteratorType;

            typedef typename LagrangePointSetType
              :: template Codim< faceCodim > :: SubEntityIteratorType
              FaceDofIteratorType;

            typedef typename DiscreteFunctionSpaceType :: DomainType DomainType ;

            const GridPartType &gridPart = space_.gridPart();

            // default is false
            bool hasDirichletBoundary = false;

            typedef typename EntityType :: Geometry Geometry;
            const Geometry& geo = entity.geometry();

            // get Lagrange pionts from space
            const LagrangePointSetType &lagrangePointSet = space_.lagrangePointSet( entity );

            // get number of Lagrange Points
            const int localBlocks = lagrangePointSet.size();

            //map local to global BlockDofs
            std::vector<size_t> globalBlockDofs(localBlocks);
            // space_.blockMapper().mapEntityDofs(entity,globalBlockDofs);
            space_.blockMapper().map(entity,globalBlockDofs);

            IntersectionIteratorType it = gridPart.ibegin( entity );
            const IntersectionIteratorType endit = gridPart.iend( entity );
            for( ; it != endit; ++it )
            {
              typedef typename IntersectionIteratorType :: Intersection IntersectionType;
              const IntersectionType& intersection = *it;

              // if intersection is with boundary, adjust data
              if( intersection.boundary() )
              {
                // get face number of boundary intersection
                const int face = intersection.indexInInside();

                // get dof iterators
                FaceDofIteratorType faceIt
                  = lagrangePointSet.template beginSubEntity< faceCodim >( face );
                const FaceDofIteratorType faceEndIt
                  = lagrangePointSet.template endSubEntity< faceCodim >( face );
                for( ; faceIt != faceEndIt; ++faceIt )
                {
                  // get local dof number (expensive operation, therefore cache result)
                  const int localBlock = *faceIt;

                  // get global coordinate of point on boundary
                  const DomainType global = geo.global( lagrangePointSet.point( localBlock ) );

                  // get dirichlet information from model
                  const bool isDirichletDof = model.isDirichletPoint( global );

                  // mark dof
                  if( isDirichletDof )
                  {
                    // mark global DoF number
                    assert( globalBlockDofs[ localBlock ] < dirichletBlocks_.size() );
                    dirichletBlocks_[globalBlockDofs[ localBlock ] ] = true ;

                    // we have Dirichlet values
                    hasDirichletBoundary = true ;
                  }
                }
              }
            }

            return hasDirichletBoundary;
          }

        //! pointer to slave dofs
        const ModelType& model_;
        const DiscreteFunctionSpaceType& space_;
        mutable std::vector< bool > dirichletBlocks_;
        mutable bool hasDirichletDofs_ ;
        mutable int sequence_ ;

        class DirichletBuilder;
    };

  template < class Model, class Space >
    class DirichletConstraints< Model,Space > :: DirichletBuilder
    : public CommDataHandleIF< DirichletBuilder, int >
    {
      public:
        typedef Space SpaceType;
        typedef typename SpaceType::BlockMapperType MapperType;

        enum { nCodim = SpaceType :: GridType :: dimension + 1 };

      public:
        typedef int DataType;

        const int myRank_;
        const int mySize_;

        typedef DirichletConstraints< Model,Space > DirichletType;
        const DirichletType &dirichlet_;

        const SpaceType &space_;
        const MapperType &mapper_;

      public:
        DirichletBuilder( const DirichletType &dirichlet,
            const SpaceType &space,
            const MapperType &mapper )
          : myRank_( space.gridPart().comm().rank() ),
          mySize_( space.gridPart().comm().size() ),
          dirichlet_( dirichlet ),
          space_( space ),
          mapper_( mapper )
      {
      }
        bool contains ( int dim, int codim ) const
        {
          return mapper_.contains( codim );
        }

        bool fixedsize ( int dim, int codim ) const
        {
          return false;
        }

        //! read buffer and apply operation
        template< class MessageBuffer, class Entity >
          inline void gather ( MessageBuffer &buffer,
              const Entity &entity ) const
          {
            unsigned int localBlocks = mapper_.numEntityDofs( entity );
            std::vector<std::size_t> globalBlockDofs(localBlocks);
            mapper_.mapEntityDofs( entity, globalBlockDofs );
            for( unsigned int localBlockDof = 0 ; localBlockDof < localBlocks; ++ localBlockDof )
            {
              if ( dirichlet_.dirichletBlocks_[ globalBlockDofs[localBlockDof] ] )
                buffer.write( 1 );
              else
                buffer.write( 0 );
            }
          }

        //! read buffer and apply operation
        //! scatter is called for one every entity
        //! several times depending on how much data
        //! was gathered
        template< class MessageBuffer, class EntityType >
          inline void scatter ( MessageBuffer &buffer,
              const EntityType &entity,
              size_t n )
          {
            unsigned int localBlocks = mapper_.numEntityDofs( entity );
            std::vector<std::size_t> globalBlockDofs(localBlocks);
            mapper_.mapEntityDofs( entity, globalBlockDofs );
            assert( n == globalBlockDofs.size() );
            assert( n == size(entity) );
            for( unsigned int localBlock = 0 ; localBlock < localBlocks; ++ localBlock )
            {
              int val;
              buffer.read(val);
              if (val == 1)
                dirichlet_.dirichletBlocks_[ globalBlockDofs[localBlock] ] = true;
            }
          }

        //! return local dof size to be communicated
        template< class Entity >
          size_t size ( const Entity &entity ) const
          {
            return mapper_.numEntityDofs( entity );
          }
    };

} // end namespace Dune
#endif
