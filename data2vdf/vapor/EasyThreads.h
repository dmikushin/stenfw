
#ifndef	_EasyThreads_h_
#define	_EasyThreads_h_
#ifndef WIN32
#include <pthread.h>
#endif
#include "MyBase.h"

namespace VetsUtil {

class COMMON_API EasyThreads : public MyBase {

public:


 EasyThreads(int nthreads);
 ~EasyThreads();
 int	ParRun(void *(*start)(void *), void **arg);
 int	Barrier();
 int	MutexLock();
 int	MutexUnlock();
 static void	Decompose(int n, int size, int rank, int *offset, int *length);
 static int	NProc();
 int	GetNumThreads() const {return(nthreads_c); }

private:
 int	nthreads_c;
#ifndef WIN32
 pthread_t	*threads_c;
 pthread_attr_t	attr_c;
 pthread_cond_t	cond_c;
 pthread_mutex_t	barrier_lock_c;
 pthread_mutex_t	mutex_lock_c;
#endif

 int	block_c;
 int	count_c;	// counters for barrier
};

};

#endif
