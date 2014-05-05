//
//      $Id: BlkMemMgr.h,v 1.7 2010/11/15 20:54:30 clynejp Exp $
//

#ifndef	_BlkMemMgr_h_
#define	_BlkMemMgr_h_

#include <vapor/MyBase.h>

namespace VAPoR {

//
//! \class BlkMemMgr
//! \brief The VetsUtil BlkMemMgr class
//! \author John Clyne
//! \version $Revision: 1.7 $
//! \date    $Date: 2010/11/15 20:54:30 $
//!
//! A block-based memory allocator. Allocates contiguous runs of
//! memory blocks from a memory pool of user defined size.
//!
//! N.B. the memory pool is stored in a static class member and
//! can only be freed by calling RequestMemSize() with a zero value 
//! after all instances of this class have been destroyed
// 
class BlkMemMgr : public VetsUtil::MyBase {

public:
 //! Initialize a memory allocator
 //
 //! Initialize a block-based memory allocator
 //
 BlkMemMgr();
 virtual ~BlkMemMgr();

 //! Alloc space from memory pool
 //
 //! Return a pointer to the specified amount of memory from the memory pool
 //! \param[in] num_blks Size of memory region requested in blocks
 //! \param[in] fill If true, the allocated memory will be cleared to zero
 //! \retval ptr A pointer to the requested memory pool
 //
 void	*Alloc(size_t num_blks, bool fill = false);

 //! Free memory
 //
 //! Frees memory previosly allocated with the \b Alloc() method.
 //! \param[in] ptr Pointer to memory returned by previous call to 
 //! \b Alloc().
 void	FreeMem(void *ptr);

 //! Set the size of the memory pool used by the memory allocator
 //
 //! Initialize a block-based memory allocator. The static memory pool
 //! is not changed until the first instance of an object of this
 //! class is created, or if there are no instances of objects the
 //! memmory pool is changed immediately. The only way to free memory
 //! from the static memory pool is to call this static method with
 //! either \p blk_size or \p num_blks set to zero. 
 //!
 //! \param[in] blk_size Size of a single memory block in bytes
 //! \param[in] num_blks Size of memory pool in blocks. This is the
 //! maximum amount that will be available through subsequent 
 //! \b Alloc() calls.
 //! \param[in] page_aligned If true, start address of memory pool 
 //! will be page aligned
 //
 static void RequestMemSize(
	size_t blk_size, size_t num_blks, int page_aligned = 1
 );

 static size_t GetBlkSize() {return(_blk_size);}

private:
 static size_t	_page_aligned_req;	// requested page align memory (boolean)
 static size_t	_mem_size_req;	// requested size of mem in blocks
 static size_t	_blk_size_req;	// requested size of block in bytes

 static size_t	_page_aligned;	// page align memory (boolean)
 static size_t	_mem_size;	// size of mem in blocks
 static size_t	_blk_size;	// size of block in bytes

 static size_t	*_free_table;	// free block table
 static unsigned char	*_blks;	// memory pool
 static unsigned char	*_blkptr;	// page-aligned memory pool
 static int _ref_count;	// # instances of object.

 int	_objInitialized;	// has the obj successfully been initalized?

 static int	_Reinit();

};
};

#endif	//	_BlkMemMgr_h_
