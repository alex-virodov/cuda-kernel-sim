#include <stdio.h>
#include <memory.h>
#include <Windows.h>
#include <list>
#include "wb.h"

tv blockDim;
tv blockIdx; 
tv threadIdx;

// A thread queue entry
struct qentry { 
	int   id;						// Thread id, used to compute blk/thread xyz
	void* fiber;					// System fiber id
	enum  { started, done } state;	// Thread state, needed for cleanup
};

std::list<qentry> queue;			// Fiber queue, first is the active one
void* fiber_main;					// The dispatch fiber

//** Start and cleanup after a user-passed function (in a closure pointed by LPVOID)
void CALLBACK q_start(LPVOID param) 
{
	// printf("entered\n");

	((tclosure*)param)->call();

	// Assuming first entry in queue
	queue.front().state = qentry::done;
	SwitchToFiber(fiber_main);
}

//** Run things in fiber queue until all are done
void run_queue()
{
	while (!queue.empty()) {
		// Compute blockidx, threadidx
		threadIdx.x = threadIdx.y = threadIdx.z = threadIdx.w = 0;
		threadIdx.x = queue.front().id % blockDim.x;
		threadIdx.y = queue.front().id / blockDim.x;

		// Execute the fiber until either returns or yields
		SwitchToFiber(queue.front().fiber);

		if (queue.front().state == qentry::done) {
			// Done, remove from queue
			DeleteFiber(queue.front().fiber);
			queue.pop_front();
		} else {
			// Reschedule
			queue.push_back(queue.front());
			queue.pop_front();
		}
	}
}

//** Schedule threads on a given block/grid size, doing
//** one block at a time.
void run_scheduler(tv& szblk, tv& szgrid, tclosure& closure)
{
	const int wrap_size = 8;

	fiber_main = ConvertThreadToFiber(0);

	blockIdx.x  = blockIdx.y  = blockIdx.z  = blockIdx.w  = 0;
	threadIdx.x = threadIdx.y = threadIdx.z = threadIdx.w = 0;

	blockDim = szblk;

	for (int j = 0; j < szgrid.y; j++) {
		for (int i = 0; i < szgrid.x; i++) {
			blockIdx.x = i;
			blockIdx.y = j;

			for (int u = 0; u < szblk.x*szblk.y; u++) {
				qentry q = { u, CreateFiber(1024, q_start, (LPVOID)&closure), qentry::started };
				//queue.push_back(q);
				queue.push_front(q); // in reverse order to not rely on order of scheduling (since this is not really parallel)
			}
			run_queue();
		}
	}
}

//** Sync between threads by yielding to others. Since threads are scheduled
//** in a consistent order, this is enough to actually sync the threads.
void __syncthreads() {
	SwitchToFiber(fiber_main);
}
