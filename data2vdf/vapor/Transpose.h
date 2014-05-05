//
//      $Id: Transpose.h,v 1.3 2010/09/24 19:16:03 clynejp Exp $
//


#ifndef __Transpose__
#define __Transpose__


namespace VetsUtil {

  //
  // blocked submatrix Transpose suitable for multithreading
  //   *a : pointer to input matrix
  //   *b : pointer to output matrix
  //    p1,p2: starting index of submatrix (row,col)
  //    m1,m2: size of submatrix (row,col)
  //    s1,s2: size of entire matrix (row,col)
  //
  
  void Transpose(float *a,float *b,int p1,int m1,int s1,int p2,int m2,int s2);

  // specialization for Real -> Complex
  // note the S1 matrix dimension is for the Real matrix
  // and the size of the Complex output is then s2 x (S1/2+1)

  
  //
  // blocked matrix Transpose single threaded
  //   *a : pointer to input matrix
  //   *b : pointer to output matrix
  //    s1,s2: size of entire matrix (row,col)
  //
  
  void Transpose(float *a,float *b,int s1,int s2);

    
  
};

#endif
