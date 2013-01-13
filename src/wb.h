#pragma once

#define __global__
#define __shared__ static

// === 4-vector type ===
class tv { 
public:
	int x, y, z, w; 
	tv(int x, int y, int z, int w) : x(x), y(y), z(z), w(w) {};
	tv() : x(0), y(0), z(0), w(0) {};
};

// === dim3 ===
class dim3 : public tv {
public:
	dim3(int x, int y = 0, int z = 0) : tv(x,y,z,0) {};
};

// === "Per thread" global variables ===
extern tv blockIdx;
extern tv blockDim;
extern tv threadIdx;

// === Scheduling ===
class tclosure;

void __syncthreads();
void run_scheduler(tv& szblk, tv& szgrid, tclosure& closure);


// === Closure (base) ===
// TODO: Use boost here? Didn't want to drag a big lib in.
class tclosure {
	tv szblk;
	tv szgrid;
public:
	virtual void call() = 0;

	void run() { run_scheduler(szblk, szgrid, *this); };
	tclosure& setBlockSize (int x, int y = 0, int z = 0) { szblk .x = x; szblk .y = y; szblk .z = z; szblk .w = 0; return (*this); };
	tclosure& setGridSize  (int x, int y = 0, int z = 0) { szgrid.x = x; szgrid.y = y; szgrid.z = z; szgrid.w = 0; return (*this); };

	tclosure& setBlockSize (tv pszblk)  { szblk  = pszblk;  return (*this); };
	tclosure& setGridSize  (tv pszgrid) { szgrid = pszgrid; return (*this); };
};

// === Closure(3) ===
// TODO: Make more of these (0,1,2,4,5,...), or do variable-argument templates.
template <class A, class B , class C>
class tclosure3 : public tclosure
{

public:
	typedef void (*tfunc)(A, B, C);
	tfunc func;
	A a; B b; C c;

	tclosure3(tfunc func, A a, B b, C c) : func(func), a(a), b(b), c(c) {};

	virtual void call() { func(a,b,c); }
};

template <class A, class B , class C>
tclosure3<A, B, C> schedule(typename tclosure3<A,B,C>::tfunc func, A a, B b, C c) {
	return tclosure3<A,B,C>(func, a, b, c);
}

