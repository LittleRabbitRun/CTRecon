/**
 * Author: Rui Liu
 */

// Utilitiles functions and classes for projection and backprojection

#ifndef __UTILITIES_CUH__
#define __UTILITIES_CUH__

//Necessary header files
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include <vector>
#include <fstream>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>

#include "omp.h"



/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#if DEBUG
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
// Same function as CUDA_CHECK_RETURN
#define CUDA_SAFE_CALL(call) do{ cudaError_t err = call; if (cudaSuccess != err) {  fprintf (stderr, "Cuda error in file '%s' in line %i : %s.", __FILE__, __LINE__, cudaGetErrorString(err) );  exit(EXIT_FAILURE);  } } while (0)
#else
#define CUDA_CHECK_RETURN(value) {value;}
#define CUDA_SAFE_CALL(value) {value;}
#endif

#ifndef nullptr
#define nullptr NULL
#endif

// Use iterator to finish the 1D transversal work
#define USING_CPP_ITERATOR 1


// PI related
#ifndef __PI__
#define __PI__
#define PI 3.14159265358979323846264
#define PI_2 1.57079632679
#define PI_4 0.785398163397448
#define PI_3_4 2.356194490192344
#define PI_5_4 3.926990816987240
#define PI_7_4 5.497787143782135
#define TWOPI 6.283185307179586
#endif

#ifndef __SQUAREROOT_OF_2_OVER_2__
#define __SQUAREROOT_OF_2_OVER_2__ 0.70710678118654752440084436210485
#endif

#define FORCEINLINE 1
#if FORCEINLINE
#define INLINE __forceinline__
#else 
#define INLINE inline
#endif


typedef unsigned char byte;
typedef thrust::device_vector<float> d_vec_t;
typedef thrust::host_vector<float> h_vec_t;


// Operator overload for vector datatype

INLINE __host__ __device__ const float2 operator/(const float2& a, float b)
{
	return make_float2(a.x / b ,a.y / b);
}

INLINE __host__ __device__ const float3 operator+(const float3& a,const float3& b)
{
	return make_float3(a.x + b.x ,a.y + b.y, a.z + b.z);
}

INLINE __host__ __device__ const float3 operator-(const float3& a,const float3& b)
{
	return make_float3(a.x - b.x ,a.y - b.y, a.z - b.z);
}
INLINE __host__ __device__ const double3 operator-(const double3& a,const double3& b)
{
	return make_double3(a.x - b.x ,a.y - b.y, a.z - b.z);
}

INLINE __host__ __device__ const float3 operator*(const float3& a,const float3& b)
{
	return make_float3(a.x * b.x ,a.y * b.y, a.z * b.z);
}

INLINE __host__ __device__ const float3 operator*(const float3& a, float b)
{
	return make_float3(a.x * b ,a.y * b, a.z * b);
}

INLINE __host__ __device__ const float3 operator/(const float3& a,const float3& b)
{
	return make_float3(a.x / b.x ,a.y / b.y, a.z / b.z);
}

INLINE __host__ __device__ const float3 operator/(const float3& a, float b)
{
	return make_float3(a.x / b ,a.y / b, a.z / b);
}

INLINE __host__ __device__ const double3 operator/(const double3& a, double b)
{
	return make_double3(a.x / b ,a.y / b, a.z / b);
}


INLINE __host__ __device__ float3 operator-(const int3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

INLINE __host__ __device__ float length(const float2& a)
{
	return sqrtf(a.x * a.x + a.y * a.y);
}
INLINE __host__ __device__ float length(const float3& a)
{
	return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

INLINE __host__ __device__ double length(const double3& a)
{
	return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

INLINE __host__ __device__ const float2 normalize(const float2& a)
{
	return a / length(a);
}

INLINE __host__ __device__ const float3 normalize(const float3& a)
{
	return a / length(a);
}

INLINE __host__ __device__ const double3 normalize(const double3& a)
{
	return a / length(a);
}


INLINE __host__ __device__  float fminf(const float2& a)
{
	return fminf(a.x, a.y);
}

INLINE __host__ __device__  float fminf(const float3& a)
{
	return fminf(a.x,fminf(a.y,a.z));
}

INLINE __host__ __device__  float fmax(const float2& a)
{
	return fmaxf(a.x, a.y);
}

INLINE __host__ __device__  float fmaxf(const float3& a)
{
	return fmaxf(a.x,fminf(a.y,a.z));
}

INLINE __host__ __device__ const float3 fminf(const float3& a, const float3& b)
{
	return make_float3(fminf(a.x,b.x),fminf(a.y,b.y),fminf(a.z,b.z));
}

INLINE __host__ __device__ const float3 fmaxf(const float3& a, const float3& b)
{
	return make_float3(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z));
}

INLINE __host__ __device__ const float2 fminf(const float2& a, const float2& b)
{
	return make_float2(fminf(a.x,b.x),fminf(a.y,b.y));
}

INLINE __host__ __device__ const float2 fmaxf(const float2& a, const float2& b)
{
	return make_float2(fmaxf(a.x,b.x),fmaxf(a.y,b.y));
}


template <typename T>
__host__ __device__
INLINE T lerp(T v0, T v1, T t) {
	return fma(t, v1, fma(-t, v0, v0));
}

template<typename T>
__host__ __device__
INLINE T bilerp(T v0, T v1, T v2, T v3, T t1, T t2)
{
	T vv0 = fma(t1,v1, fma(-t1,v0,v0));
	T vv1 = fma(t1,v3, fma(-t1,v2,v2));
	return fma(t2,vv1, fma(-t2,vv0,vv0));

}

INLINE __device__  double bilerp(int2 v0, int2 v1, int2 v2, int2 v3, float t1, float t2)
{
	double v0_ = __hiloint2double(v0.y,v0.x);
	double v1_ = __hiloint2double(v1.y,v1.x);
	double v2_ = __hiloint2double(v2.y,v2.x);
	double v3_ = __hiloint2double(v3.y,v3.x);
	double vv0 = v0_ * (1-t1) + v1_ * t1;// fma(t1,v1, fma(-t1,v0,v0));
	double vv1 = v2_ * (1-t1) + v3_ * t1;// fma(t1,v3, fma(-t1,v2,v2));
	return vv0 * (1-t2) + vv1 * t2;//fma(t2,vv1, fma(-t2,vv0,vv0));
}

INLINE __device__  double bilerp(int2 v0, int2 v1, int2 v2, int2 v3, double t1, double t2)
{
	double v0_ = __hiloint2double(v0.y,v0.x);
	double v1_ = __hiloint2double(v1.y,v1.x);
	double v2_ = __hiloint2double(v2.y,v2.x);
	double v3_ = __hiloint2double(v3.y,v3.x);
	double vv0 = v0_ * (1-t1) + v1_ * t1;// fma(t1,v1, fma(-t1,v0,v0));
	double vv1 = v2_ * (1-t1) + v3_ * t1;// fma(t1,v3, fma(-t1,v2,v2));
	return vv0 * (1-t2) + vv1 * t2;//fma(t2,vv1, fma(-t2,vv0,vv0));
}


// Regularize the angle to [0, 2pi)
template<typename T>
INLINE __host__ __device__ T regularizeAngle(T curang)
{
	T c = curang;
	while(c >= TWOPI){c -= TWOPI; }
	while(c < 0){c += TWOPI; }
	return c;
}

// inversely rotate the voxel back to its 'virtual' initial position to calculate the intersection point on the 
// detector for backprojection 
INLINE __host__ __device__ void invRotVox(
		const float3& curVox, //current voxel position 
		float3& virVox, // virtual voxel initial position
		const float2& cossinT, //cosine and sine 
		const float zP) // zshift
{
	virVox.x = curVox.x * cossinT.x + curVox.y * cossinT.y;
	virVox.y =-curVox.x * cossinT.y + curVox.y * cossinT.x;
	virVox.z = curVox.z - zP;
}
INLINE __device__ float3 invRot(const float3 inV,const float2 cossin, const float zP)
{
	float3 outV;
	outV.x = inV.x * cossin.x + inV.y * cossin.y;
	outV.y =-inV.x * cossin.y + inV.y * cossin.x;
	outV.z = inV.z - zP;
	return outV;
}

namespace CTMBIR
{

	//Calculate the constant values for backprojection
	template<typename T>
	struct ConstantForBackProjection
	{
		typedef thrust::tuple<T, T> InTuple;

		ConstantForBackProjection(const T _x0, const T _y0, const T _z0):x0(_x0),y0(_y0),z0(_z0){}

		T x0;
		T y0;
		T z0;

		__device__ float3 operator()(const InTuple& tp)
		{
			T curang = regularizeAngle(thrust::get<0>(tp));
			T zP = thrust::get<1>(tp);
			T cosT = cosf(curang);
			T sinT = sinf(curang);

			return make_float3(cosT,sinT,zP);
		}
	};
	
	template<>
	struct ConstantForBackProjection<double>
	{
		typedef thrust::tuple<double, double> InTuple;

		ConstantForBackProjection(const double _x0, const double _y0, const double _z0):x0(_x0),y0(_y0),z0(_z0){}

		double x0;
		double y0;
		double z0;

		__device__ double3 operator()(const InTuple& tp)
		{
			double curang = regularizeAngle(thrust::get<0>(tp));
			double zP = thrust::get<1>(tp);
			double cosT = cos(curang);
			double sinT = sin(curang);

			return make_double3(cosT,sinT,zP);
		}
	};

}


//Calculate the detector boundary coordinates according to the center coordinates
template<typename T>
void DD3Boundaries(int nrBoundaries,
			 T *pCenters,
			 T *pBoundaries)
{
  int i;
  if (nrBoundaries >= 3)
    {
    *pBoundaries++ = 1.5 * *pCenters - 0.5 * *(pCenters+1);
    for (i=1 ; i<=(nrBoundaries-2) ; i++)
      {
               *pBoundaries++ = 0.5 * *pCenters + 0.5 * *(pCenters+1);
               pCenters++;
      }
    *pBoundaries = 1.5 * *pCenters - 0.5 * *(pCenters-1);
    }
  else
    {
      *pBoundaries = *pCenters-0.5;
      *(pBoundaries+1) = *pCenters+0.5;
    }
}


#endif
