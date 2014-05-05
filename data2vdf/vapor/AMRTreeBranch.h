//
//      $Id: AMRTreeBranch.h,v 1.8.2.1 2011/12/01 17:29:05 clynejp Exp $
//
//***********************************************************************
//                                                                      *
//                      Copyright (C)  2006	                        *
//          University Corporation for Atmospheric Research             *
//                      All Rights Reserved                             *
//                                                                      *
//***********************************************************************
//
//	File:		AMRTreeBranch.h
//
//	Author:		John Clyne
//			National Center for Atmospheric Research
//			PO 3000, Boulder, Colorado
//
//	Date:		Thu Jan 5 16:59:42 MST 2006
//
//	Description:	
//
//
#ifndef	_AMRTreeBranch_h_
#define	_AMRTreeBranch_h_

#include <vector>
#include <vapor/common.h>
#include <vapor/MyBase.h>
#include <vapor/EasyThreads.h>
#include <vapor/XmlNode.h>


namespace VAPoR {

//
//! \class AMRTreeBranch
//! \brief This class manages an octree data structure
//! \author John Clyne
//! \version $Revision: 1.8.2.1 $
//! \date    $Date: 2011/12/01 17:29:05 $
//!
//! This class manages a branch of Adaptive Mesh Refinement tree data 
//! structure. 
//! In actuality, the tree data structure is simply an octree. There is
//! not much particular about it with regard to AMR. The root of the
//! tree has a single node (cell). Nodes are refined by subdivision into 
//! octants. Each node has an integer cell identifier.  
//! Cell id's are ordered sequentially based on creation.
//! The root node has id 0 (zero). The root's children
//! are numbered 1-8, with child #1 coresponding to the octant with lowest
//! coordinate bounds and child #8 coresponding to the octant with the highest
//! coordinate bounds. 
//!
//! This class is derived from the MyBase base
//! class. Hence all of the methods make use of MyBase's
//! error reporting capability - the success of any method
//! (including constructors) can (and should) be tested
//! with the GetErrCode() method. If non-zero, an error
//! message can be retrieved with GetErrMsg().
//!
//!
//! No field data is stored with the AMRTreeBranch class.
//

class AMRTreeBranch : public VetsUtil::MyBase {

public:

 typedef long cid_t;

 //
 // Tag names for data stored in the XML tree
 //
 static const string _rootTag;			// root of the XML tree
 static const string _refinementLevelTag;	// max refinement depth of branch
 // List of parent cells in breath first traversal order
 static const string _parentTableTag;


 //
 // Attribute names for data stored in the XML tree
 //
 static const string _minExtentsAttr;	// min coord bounds of branch
 static const string _maxExtentsAttr;	// max coord bounds of branch
 static const string _locationAttr;		// toplogical coords of branch


 //! Constructor for the AMRTreeBranch class.
 //!
 //! Contructs a root tree with no children. The refinement level is zero (0)
 //! \param[in] parent Parent node of XML tree
 //! \param[in] min A three element array indicating the minimum X,Y,Z bounds 
 //! of the tree branch, specified in user coordinates.
 //! \param[in] max A three element array indicating the maximum X,Y,Z bounds 
 //! of the tree branch, specified in user coordinates.
 //! \param[in] location A three element integer array indicating the
 //! coordinates of this branch relative to other branches in an AMR tree.
 //!
 //! \sa AMRTree
 //
 AMRTreeBranch(
	XmlNode *parent,
	const double min[3],
	const double max[3],
	const size_t location[3]
 );

 virtual ~AMRTreeBranch();

 void Update();

 //! Delete a cell from the tree branch
 //!
 //! Deletes the indicated cell from the tree branch
 //! \param[in] cellid Cell id of cell to delete
 //!
 //! \retval status Returns a non-negative value on success
 //
 int	DeleteCell(cid_t cellid);

 //! Find a cell containing a point.
 //!
 //! This method returns the cell id of the node containing a point.
 //! The coordinates of the point are specfied in the user coordinates
 //! defined by the during the branch's construction (it is assumed that
 //! the branch occupies Cartesian space and that all cells are rectangular.
 //! By default, the leaf node containing the point is returned. However,
 //! an ancestor node may be returned by specifying limiting the refinement 
 //! level.
 //!
 //! \param[in] ucoord A three element array containing the coordintes
 //! of the point to be found. 
 //! \param[in] ref_level The refinement level of the cell to be
 //! returned. If -1, a leaf node (cell) is returned.
 //! \retval cellid A valid cell id is returned if the branch contains 
 //! the indicated point. Otherwise a negative int is returned.
 //
 cid_t	FindCell(const double ucoord[3], int ref_level = -1) const;

 bool ValidCell(cid_t cellid) const {
	return(cellid < _octree->get_num_cells(_octree->get_max_level())); 
 };

 //! Return the user coordinate bounds of a cell
 //!
 //! This method returns the minimum and maximum extents of the 
 //! indicated cell. The extents are in user coordinates as defined
 //! by the constructor for the class. The bounds for the root cell
 //! are guaranteed to be the same as those used to construct the class.
 //!
 //! \param[in] cellid The cell id of the cell whose bounds are to be returned
 //! \param[out] minu A three element array to which the minimum cell 
 //! extents will be copied.
 //! \param[out] maxu A three element array to which the maximum cell 
 //! extents will be copied.
 //! \retval status Returns a non-negative value on success
 //!
 int	GetCellBounds(
	cid_t cellid, double minu[3], double maxu[3]
 ) const;

 //! Return the topological coordinates of a cell within the branch.
 //!
 //! This method returns topological coordinates of a cell within
 //! the tree, relative to the cell's refinement level. Each cell
 //! in a branch has an i-j-k location for it's refinement level. The range
 //! of i-j-k values runs from \e min to \e max, where \e min is given by
 //! location * 2^j, where \e location is provided by the \p location
 //! parameter of the constructor and \e j is the refinement level
 //! of the cell. The value of \e max is \min + 2^j - 1, where \e j is
 //! again the cell's refinement level. Hence, the cell's location
 //! is relative to the branch's location.
 //!
 //! \param[in] cellid The cell id of the cell whose bounds are to be returned
 //! \param[out] xyz The cell's location
 //! \param[out] reflevel The refinment level of the cell
 //!
 //! \retval status Returns a non-negative value on success.  A negative
 //! int is returned if \p cellid does not exist in the tree.
 //!
 int	GetCellLocation(cid_t cellid, size_t xyz[3],int *reflevel) const;


 //! Return the cellid for a cell with the given i-j-k coordinates
 //!
 //! This method returns the cellid for the cell indicated by
 //! the i-j-k coordinates at a given refinement level.
 //!
 //! \param[in] xyz The cell's location 
 //! \param[in] reflevel The refinement level of the cell
 //! \param[in] cellid The cell id of the cell whose bounds are to be returned
 //!
 //! \retval status Returns a non-negative value on success. A negative
 //! int is returned \p xyz are invalid for refinement level.
 //!
 //! \sa GetCellLocation()
 //!
 AMRTreeBranch::cid_t    GetCellID(const size_t xyz[3], int reflevel) const;


 //! Returns the cell id of the first child of a node
 //!
 //! This method returns the cell id of the first of the eight children
 //! of the cell indicated by \p cellid. The remaining seven children's
 //! ids may be calulated by incrementing the first child's id successively.
 //! I.e. the children have contiguous integer ids. Octants are ordered
 //! with the X axis varying fastest, followed by Y, then Z.
 //!
 //! \param[in] cellid The cell id of the cell whose first child is
 //! to be returned.
 //! \retval cellid A valid cell id is returned if the branch contains 
 //! the indicated point. Otherwise a negative int is returned.
 //
 cid_t	GetCellChildren(cid_t cellid) const;

 //! Return the refinement level of a cell
 //!
 //! Returns the refinement level of the cell indicated by \p cellid. 
 //! The base (coarsest) refinement level is zero (0).
 //!
 //! \param[in] cellid The cell id of the cell whose refinement level is
 //! to be returned.
 //! \retval status Returns a non-negative value on success
 //
 int	GetCellLevel(cid_t cellid) const;


 //! Return the cell id of a cell's neighbor.
 //!
 //! Returns the cell id of the cell adjacent to the indicated face
 //! of the cell with id \p cellid. The \p face parameter is an integer
 //! in the range [0..5], where 0 coresponds to the face on the XY plane 
 //! with Z = 0; 1 is the YZ plane with X = 1; 2 is the XY plane, Z = 1;
 //! 3 is the YZ plane; Z = 0, 4 is the XZ plane, Y = 0; and 5 is the
 //! XZ plane, Z = 1.
 //!
 //! \param[in] cellid The cell id of the cell whose neighbor is
 //! to be returned.
 //! \param[in] face Indicates the cell's face adjacent to the desired
 //! neighbor.
 //! \retval cellid A valid cell id is returned if the branch contains 
 //! the indicated point. Otherwise a negative int is returned.
 cid_t	GetCellNeighbor(cid_t cellid, int face) const;

 //! Return number of cells in the branch
 //!
 //! Returns the total number of cells in a branch, including parent cells
 //!
 cid_t GetNumCells() const;

 //! Return number of cells in the branch
 //!
 //! Returns the total number of cells in a branch, including parent cells,
 //! up to and including the indicated refinement level.
 //!
 cid_t GetNumCells(int ref_level) const;

 //! Returns the cell id of the parent of a child node
 //!
 //! This method returns the cell id of the parent 
 //! of the cell indicated by \p cellid. 
 //!
 //! \param[in] cellid The cell id of the cell whose parent is
 //! to be returned.
 //! \retval cellid A valid cell id is returned if the branch contains 
 //! the indicated point. Otherwise a negative int is returned.
 //
 cid_t	GetCellParent(cid_t cellid) const;

 //! Returns the maximum refinement level of any cell in this branch
 //
 int	GetRefinementLevel() const {
	return ((int) (_octree->get_max_level()));
 };

 //! Returns the cellid root node
 //
 cid_t	GetRoot() const {return(0);};

 //! Returns true if the cell with id \p cellid has children. 
 //! 
 //! Returns true or false indicated whether the specified cell
 //! has children or not.
 //
 int	HasChildren(cid_t cellid) const;


 //! Refine a cell
 //!
 //! This method refines a cell, creating eight new child octants. Upon
 //! success the cell id of the first child is returned.
 //!
 //! \param[in] cellid The cell id of the cell to be refined
 //! \retval cellid A valid cell id is returned if the branch contains 
 //! the indicated point. Otherwise a negative int is returned.
 //!
 //
 AMRTreeBranch::cid_t	RefineCell(cid_t cellid);


 //! \copydoc AMRTree::EndRefinement()
 //!
 //
 void EndRefinement();

 //! \copydoc AMRTree::GetNextCell()
 //
 AMRTreeBranch::cid_t	GetNextCell(bool restart);

 //! Return the serialized offset of the block with id \param cellid
 //!
 //! This method returns the offset to the indicated cell in a breath-first
 //! traversal of the tree.
 //!
 //! \retval offset Returns a negative int of \p cellid is invalid, otherwise
 //! returns the offset of the cell
 //
 long GetCellOffset(cid_t cellid) const;

 int SetParentTable(const vector<long> &table);

private:

	//
	// cell ids indicate the order in which the cell was added to the 
	// tree via refinement. They also determine the octant that a cell 
	// occupies by virtue of the fact that refinement generates 8 new
	// cells, inserted into the tree in octant order. The root of the 
	// tree always has id==0
	//
 class octree {
 public:


	octree();
	void clear();
	cid_t refine_cell(cid_t cellid);
	void end_refinement();
	cid_t get_parent(cid_t cellid) const;

	//
	// returns the cellid of the child in the first octant. The
	// id's of the remaining children are numbered sequentially
	//
	cid_t get_children(cid_t cellid) const;

	bool has_children(cid_t cellid) const;
	// 
	// Get id of next child in a breadth first traversal of 
	// the tree. If 'restart' is true, traversal starts from the
	// top of the tree, else traversal starts from the cell returned
	// from the previous call. When the tree has been completely 
	// treversed a negative number is returned.
	//
	cid_t get_next(bool restart);

	long get_offset(cid_t cellid) const;

	// 
	// Return the octant occupied by 'cellid'. Octants are numbered
	// sequentially from 0 with X varying fastest, followed by y, than Z.
	//
	int get_octant(cid_t cellid) const;
	int get_max_level() const {return(_max_level); };


	// Get the cellid for the cell with the specified coordinates (if
	// it exists). The cartesian coordinates are respective
	// to the refinement level of the cell. Their range is from
	// 0..(2^j)-1, where j is the refinment level. If no cell
	// exists at (xyz, level) a negative int is returned.
	//
	cid_t get_cellid(const size_t xyz[3], int level) const;

	int get_location(cid_t cellid, size_t xyz[3], int *level) const;

	int get_level(cid_t cellid) const;

	cid_t get_num_cells(int ref_level) const;

 private:

	vector <cid_t> _bft_next;
	vector <cid_t>::iterator _bft_itr;
	vector <cid_t> _bft_children;
	vector <cid_t> _num_cells;
	int _max_level;	// maximum refinment level in tree. Base level is 0

	vector <long> _offsets; // vector of offsets to speed breath-first search 

	typedef struct {
		cid_t parent;
		cid_t child;	// id of first child. Remaining children have
						// consecutivie ids
	} _node_t;

	vector <_node_t> _tree;	// interal octree representation.

	void _init();


 };


 octree *_octree;

 size_t _location[3];    // Topological coords of branch  in blocks
 double _minBounds[3];  // Min domain extents expressed in user coordinates
 double _maxBounds[3];  // Max domain extents expressed in user coordinates

 XmlNode	*_rootNode;	// root of XML tree used to represent AMR tree


};

};

#endif	//	_AMRTreeBranch_h_
