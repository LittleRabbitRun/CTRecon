
#include "DD_MultiGPU_ker.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <omp.h>

#define BACK_BLKX 64
#define BACK_BLKY 4
#define BACK_BLKZ 1
#define BLKX 32
#define BLKY 8
#define BLKZ 1


#ifndef __PI__
#define __PI__
#define PI		3.141592653589793
#define PI_2		1.570796326794897
#define PI_4		0.785398163397448
#define PI_3_4		2.356194490192344
#define PI_5_4		3.926990816987241
#define PI_7_4		5.497787143782138
#define TWOPI       6.283185307179586
#endif

#define FORCEINLINE 1
#if FORCEINLINE
#define INLINE __forceinline__
#else
#define INLINE inline
#endif

#ifndef DEBUG
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



typedef unsigned char byte;



#ifndef nullptr
#define nullptr NULL
#endif

INLINE __host__ __device__ const float2 operator/(const float2& a, float b)
{
	return make_float2(a.x / b, a.y / b);
}

INLINE __host__ __device__ const float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

INLINE __host__ __device__ const float3 operator-(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}


INLINE __host__ __device__ const float2 operator-(const float2& a, const float2& b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}

INLINE __host__ __device__ const float3 operator*(const float3& a, const float3& b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

INLINE __host__ __device__ const float3 operator*(const float3& a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

INLINE __host__ __device__ const float3 operator/(const float3& a, const float3& b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

INLINE __host__ __device__ const float3 operator/(const float3& a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}


INLINE __host__ __device__ const double3 operator/(const double3& a, double b)
{
	return make_double3(a.x / b, a.y / b, a.z / b);
}


INLINE __host__ __device__ const float3 operator-(const int3& a, const float3& b)
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

INLINE __host__ __device__ float fminf(const float2& a)
{
	return fminf(a.x, a.y);
}

INLINE __host__ __device__ float fminf(const float3& a)
{
	return fminf(a.x, fminf(a.y, a.z));
}

INLINE __host__ __device__ float fmaxf(const float2& a)
{
	return fmaxf(a.x, a.y);
}

INLINE __host__ __device__ float fmaxf(const float3& a)
{
	return fmaxf(a.x, fmaxf(a.y, a.z));
}

INLINE __host__ __device__ const float3 fminf(const float3& a, const float3& b)
{
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

INLINE __host__ __device__ const float3 fmaxf(const float3& a, const float3& b)
{
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

INLINE __host__ __device__ const float2 fminf(const float2& a, const float2& b)
{
	return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

INLINE __host__ __device__ const float2 fmaxf(const float2& a, const float2& b)
{
	return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}


INLINE __host__ __device__ bool intersectBox(
	const float3& sour,
	const float3& dir,
	const float3& boxmin,
	const float3& boxmax,
	float* tnear, float* tfar)
{
	const float3 invR = make_float3(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
	const float3 tbot = invR * (boxmin - sour);
	const float3 ttop = invR * (boxmax - sour);

	const float3 tmin = fminf(ttop, tbot);
	const float3 tmax = fmaxf(ttop, tbot);

	const float largest_tmin = fmaxf(tmin);
	const float smallest_tmax = fminf(tmax);
	*tnear = largest_tmin;
	*tfar = smallest_tmax;
	return smallest_tmax > largest_tmin;
}

template<typename T>
INLINE __host__ __device__ T regularizeAngle(T curang)
{
	T c = curang;
	while (c >= TWOPI){ c -= TWOPI; }
	while (c < 0){ c += TWOPI; }
	return c;
}


INLINE __host__ __device__ void invRotVox(
	const float3& curVox,
	float3& virVox,
	const float2& cossinT,
	const float zP)
{
	virVox.x = curVox.x * cossinT.x + curVox.y * cossinT.y;
	virVox.y =-curVox.x * cossinT.y + curVox.y * cossinT.x;
	virVox.z = curVox.z - zP;
}

INLINE __device__ float3 invRot(
	const float3 inV,
	const float2 cossin,
	const float zP)
{
	float3 outV;
	outV.x = inV.x * cossin.x + inV.y * cossin.y;
	outV.x =-inV.x * cossin.y + inV.y * cossin.x;
	outV.z = inV.z - zP;
	return outV;
}


namespace CTMBIR
{

	struct ConstantForBackProjection4{

		float x0;
		float y0;
		float z0;

		typedef thrust::tuple<float, float> InTuple;
		ConstantForBackProjection4(const float _x0, const float _y0, const float _z0)
			: x0(_x0), y0(_y0), z0(_z0){}

		__device__ float3 operator()(const InTuple& tp)
		{
			float curang = regularizeAngle(thrust::get<0>(tp));
			float zP = thrust::get<1>(tp);
			float cosT = cosf(curang);
			float sinT = sinf(curang);
			return make_float3(cosT, sinT, zP);
		}

	};

}


template<typename T>
void DD3Boundaries(int nrBoundaries, T*pCenters, T *pBoundaries)
{
	int i;
	if (nrBoundaries >= 3)
	{
		*pBoundaries++ = 1.5 * *pCenters - 0.5 * *(pCenters + 1);
		for (i = 1; i <= (nrBoundaries - 2); i++)
		{
			*pBoundaries++ = 0.5 * *pCenters + 0.5 * *(pCenters + 1);
			pCenters++;
		}
		*pBoundaries = 1.5 * *pCenters - 0.5 * *(pCenters - 1);
	}
	else
	{
		*pBoundaries = *pCenters - 0.5;
		*(pBoundaries + 1) = *pCenters + 0.5;
	}

}

template<typename T>
void DD3Boundaries(int nrBoundaries, std::vector<T>& pCenters, T *pBoundaries)
{
	int i;
	if (nrBoundaries >= 3)
	{
		*pBoundaries++ = 1.5 * *pCenters - 0.5 * *(pCenters + 1);
		for (i = 1; i <= (nrBoundaries - 2); i++)
		{
			*pBoundaries++ = 0.5 * *pCenters + 0.5 * *(pCenters + 1);
			pCenters++;
		}
		*pBoundaries = 1.5 * *pCenters - 0.5 * *(pCenters - 1);
	}
	else
	{
		*pBoundaries = *pCenters - 0.5;
		*(pBoundaries + 1) = *pCenters + 0.5;
	}

}

template<typename T>
void DD3Boundaries(int nrBoundaries,T *pCenters, std::vector<T>& pB)
{
	T* pBoundaries = &pB[0];
	int i;
	if (nrBoundaries >= 3)
	{
		*pBoundaries++ = 1.5 * *pCenters - 0.5 * *(pCenters + 1);
		for (i = 1; i <= (nrBoundaries - 2); i++)
		{
			*pBoundaries++ = 0.5 * *pCenters + 0.5 * *(pCenters + 1);
			pCenters++;
		}
		*pBoundaries = 1.5 * *pCenters - 0.5 * *(pCenters - 1);
	}
	else
	{
		*pBoundaries = *pCenters - 0.5;
		*(pBoundaries + 1) = *pCenters + 0.5;
	}

}

template<typename T>
void DD3Boundaries(int nrBoundaries,std::vector<T>& pC, std::vector<T>& pB)
{
	T* pCenters = &pC[0];
	T* pBoundaries = &pB[0];
	int i;
	if (nrBoundaries >= 3)
	{
		*pBoundaries++ = 1.5 * *pCenters - 0.5 * *(pCenters + 1);
		for (i = 1; i <= (nrBoundaries - 2); i++)
		{
			*pBoundaries++ = 0.5 * *pCenters + 0.5 * *(pCenters + 1);
			pCenters++;
		}
		*pBoundaries = 1.5 * *pCenters - 0.5 * *(pCenters - 1);
	}
	else
	{
		*pBoundaries = *pCenters - 0.5;
		*(pBoundaries + 1) = *pCenters + 0.5;
	}

}



///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
// Get one sub-volume from the whole volume.
// Assume that the volumes are stored in Z, X, Y order
template<typename T>
void getSubVolume(const T* vol,
		const size_t XN, const size_t YN, const size_t ZN,
		const size_t ZIdx_Start, const size_t ZIdx_End, T* subVol)
{
	const size_t SZN = ZIdx_End - ZIdx_Start;
	for (size_t yIdx = 0; yIdx != YN; ++yIdx)
	{
		for (size_t xIdx = 0; xIdx != XN; ++xIdx)
		{
			for (size_t zIdx = ZIdx_Start; zIdx != ZIdx_End; ++zIdx)
			{
				subVol[(yIdx * XN + xIdx) * SZN + (zIdx - ZIdx_Start)] = vol[(yIdx * XN + xIdx) * ZN + zIdx];
			}
		}
	}
}

template<typename T>
void getSubVolume(const T* vol,
		const size_t XYN, const size_t ZN,
		const size_t ZIdx_Start, const size_t ZIdx_End, T* subVol)
{
	const size_t SZN = ZIdx_End - ZIdx_Start;
	for (size_t xyIdx = 0; xyIdx != XYN; ++xyIdx)
	{
		for (size_t zIdx = ZIdx_Start; zIdx != ZIdx_End; ++zIdx)
		{
			subVol[xyIdx * SZN + (zIdx - ZIdx_Start)] = vol[xyIdx * ZN + zIdx];
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////

// For projection, before we divide the volume into serveral sub-volumes, we have
// to calculate the Z index range
template<typename T>
void getVolZIdxPair(const thrust::host_vector<T>& zPos, // Z position of the source.
		//NOTE: We only assume the spiral CT case that zPos is increasing.
		const size_t PrjIdx_Start, const size_t PrjIdx_End,
		const T detCntIdxV, const T detStpZ, const int DNV,
		const T objCntIdxZ,	const T dz, const int ZN, // Size of the volume
		int& ObjIdx_Start, int& ObjIdx_End) // The end is not included
{
	const T lowerPart = (detCntIdxV + 0.5) * detStpZ;
	const T upperPart = DNV * detStpZ - lowerPart;
	const T startPos = zPos[PrjIdx_Start] - lowerPart;
	const T endPos = zPos[PrjIdx_End - 1] + upperPart;

	ObjIdx_Start = floor((startPos / dz) + objCntIdxZ - 1);
	ObjIdx_End = ceil((endPos / dz) + objCntIdxZ + 1) + 1;

	ObjIdx_Start = (ObjIdx_Start < 0) ? 0 : ObjIdx_Start;
	ObjIdx_Start = (ObjIdx_Start > ZN) ? ZN : ObjIdx_Start;

	ObjIdx_End = (ObjIdx_End < 0) ? 0 : ObjIdx_End;
	ObjIdx_End = (ObjIdx_End > ZN) ? ZN : ObjIdx_End;
}

///////////////////////////////////////////////////////////////////////////////////
// For backprojection, after decide the subvolume range, we have to decide the
// projection range to cover the subvolume.
template<typename T>
void getPrjIdxPair(const thrust::host_vector<T>& zPos, // Z Position of the source.
		// NOTE: we assume that it is pre sorted
		const size_t ObjZIdx_Start, const size_t ObjZIdx_End, // sub vol range,
		// NOTE: the objZIdx_End is not included
		const T objCntIdxZ, const T dz, const int ZN,
		const T detCntIdxV, const T detStpZ, const int DNV,
		int& prjIdx_Start, int& prjIdx_End)
{
	const int PN = zPos.size();

	const T lowerPartV = (ObjZIdx_Start - objCntIdxZ - 0.5) * dz;
	const T highrPartV = lowerPartV + (ObjZIdx_End - ObjZIdx_Start) * dz;

	const T lowerPartDet = (detCntIdxV + 0.5) * detStpZ;
	const T upperPartDet = DNV * detStpZ - lowerPartDet;

	//The source position
	const T sourLPos = lowerPartV - upperPartDet;
	const T sourHPos = highrPartV + lowerPartDet;

	prjIdx_Start = thrust::upper_bound(zPos.begin(),zPos.end(),sourLPos) - zPos.begin() - 1;
	prjIdx_End = thrust::upper_bound(zPos.begin(),zPos.end(),sourHPos) - zPos.begin() + 2;
	prjIdx_Start = (prjIdx_Start < 0) ? 0 : prjIdx_Start;
	prjIdx_Start = (prjIdx_Start > PN)? PN: prjIdx_Start;

	prjIdx_End = (prjIdx_End < 0) ? 0 : prjIdx_End;
	prjIdx_End = (prjIdx_End > PN) ? PN : prjIdx_End;
}


////////////////////////////////////////////////////////////////////////////////////
// The volume is also stored in Z, X, Y order
// Not tested yet.
template<typename T>
void combineVolume(
	T* vol, // The volume to be combined
	const int XN, const int YN, const int ZN,
	T** subVol, // All sub volumes
	const int* SZN, // Number of slices for each subVolume
	const int subVolNum) // Number of sub volumes
{
	int kk = 0;
	for (size_t yIdx = 0; yIdx != YN; ++yIdx)
	{
		for (size_t xIdx = 0; xIdx != XN; ++xIdx)
		{
			kk = 0;
			for (size_t volIdx = 0; volIdx != subVolNum; ++volIdx)
			{
				for (size_t zIdx = 0; zIdx != SZN[volIdx]; ++zIdx)
				{
					vol[(yIdx * XN + xIdx) * ZN + kk] = subVol[volIdx][(yIdx * XN + xIdx) * SZN[volIdx] + zIdx];
					kk = kk + 1;
				}
			}
		}
	}
}

template<typename T>
void combineVolume(
	T* vol, // The volume to be combined
	const int XN, const int YN, const int ZN,
	thrust::host_vector<thrust::host_vector<float> >& subVol, // All sub volumes
	const int* SZN, // Number of slices for each subVolume
	const int subVolNum) // Number of sub volumes
{
	int kk = 0;
	for (size_t yIdx = 0; yIdx != YN; ++yIdx)
	{
		for (size_t xIdx = 0; xIdx != XN; ++xIdx)
		{
			kk = 0;
			for (size_t volIdx = 0; volIdx != subVolNum; ++volIdx)
			{
				for (size_t zIdx = 0; zIdx != SZN[volIdx]; ++zIdx)
				{
					vol[(yIdx * XN + xIdx) * ZN + kk] = subVol[volIdx][(yIdx * XN + xIdx) * SZN[volIdx] + zIdx];
					kk = kk + 1;
				}
			}
		}
	}
}


template<typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b)
{
	std::vector<T> res(a.size(), 0);
	std::transform(a.begin(),a.end(),b.begin(), res.begin(), [](T aa, T bb){return aa - bb;});
	return res;
}



template<typename T>
__device__ inline T intersectLength(const T& fixedmin, const T& fixedmax, const T& varimin, const T& varimax)
{
	const T left = (fixedmin > varimin) ? fixedmin : varimin;
	const T right = (fixedmax < varimax) ? fixedmax : varimax;
	return abs(right - left) * static_cast<double>(right > left);
}



template<typename Ta, typename Tb>
__global__ void naive_copyToTwoVolumes(Ta* in_ZXY,
	Tb* out_ZXY, Tb* out_ZYX,
	int XN, int YN, int ZN)
{
	int idz = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = threadIdx.y + blockIdx.y * blockDim.y;
	int idy = threadIdx.z + blockIdx.z * blockDim.z;
	if (idx < XN && idy < YN && idz < ZN)
	{
		int i = (idy * XN + idx) * ZN + idz;
		int ni = (idy * (XN + 1) + (idx + 1)) * (ZN + 1) + idz + 1;
		int nj = (idx * (YN + 1) + (idy + 1)) * (ZN + 1) + idz + 1;

		out_ZXY[ni] = in_ZXY[i];
		out_ZYX[nj] = in_ZXY[i];
	}
}

template<typename Ta, typename Tb>
__global__ void naive_herizontalIntegral(Ta* in, Tb* out, int N, int ZN)
{
	int zi = threadIdx.x + blockIdx.x * blockDim.x;
	if (zi < ZN)
	{
		out[zi] = in[zi];
		for (int i = 1; i < N; ++i)
		{
			out[i * ZN + zi] = out[(i - 1) * ZN + zi]
				+ in[i * ZN + zi];
		}
	}
}

template<typename Ta, typename Tb>
__global__ void naive_verticalIntegral(Ta* in, Tb* out, int N, int ZN)
{
	int xyi = threadIdx.x + blockIdx.x * blockDim.x;
	if (xyi < N)
	{
		out[xyi * ZN] = in[xyi * ZN];
		for (int ii = 1; ii < ZN; ++ii)
		{
			out[xyi * ZN + ii] = out[xyi * ZN + ii - 1]
				+ in[xyi * ZN + ii];
		}

	}
}


template<typename T>
__global__ void verticalIntegral(T* prj, int ZN, int N)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N)
	{
		int currentHead = idx * ZN;
		for (int ii = 1; ii < ZN; ++ii)
		{
			prj[currentHead + ii] = prj[currentHead + ii] + prj[currentHead + ii - 1];
		}
	}
}

template<typename T>
__global__ void horizontalIntegral(T* prj, int DNU, int DNV, int PN)
{
	int idv = threadIdx.x + blockIdx.x * blockDim.x;
	int pIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if (idv < DNV && pIdx < PN)
	{
		int headPtr = pIdx * DNU * DNV + idv;
		for (int ii = 1; ii < DNU; ++ii)
		{
			prj[headPtr + ii * DNV] = prj[headPtr + ii * DNV] + prj[headPtr + (ii - 1) * DNV];
		}
	}
}





__global__ void naive_vertialIntegral(double* in, int2* out, int N, int ZN)
{
	int xyi = threadIdx.x + blockIdx.x * blockDim.x;
	if (xyi < N)
	{
		double temp = in[xyi * ZN];
		out[xyi * ZN] = make_int2(__double2loint(temp), __double2hiint(temp));
		double temp2 = 0;
		for (int ii = 0; ii < ZN; ++ii)
		{
			temp2 = temp + in[xyi * ZN + ii];
			out[xyi * ZN + ii] = make_int2(__double2loint(temp2), __double2hiint(temp2));
			temp = temp2;
		}
	}
}



__global__ void verticalIntegral(float* prj, int ZN, int N)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N)
	{
		int currentHead = idx * ZN;
		for (int ii = 1; ii < ZN; ++ii)
		{
			prj[currentHead + ii] = prj[currentHead + ii] + prj[currentHead + ii - 1];
		}
	}
}



__global__ void horizontalIntegral(float* prj, int DNU, int DNV, int PN)
{
	int idv = threadIdx.x + blockIdx.x * blockDim.x;
	int pIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if (idv < DNV && pIdx < PN)
	{
		int headPrt = pIdx * DNU * DNV + idv;
		for (int ii = 1; ii < DNU; ++ii)
		{
			prj[headPrt + ii * DNV] = prj[headPrt + ii * DNV] + prj[headPrt + (ii - 1) * DNV];
		}
	}
}

__global__  void DD3_gpu_proj_branchless_sat2d_ker(
	cudaTextureObject_t volTex1,
	cudaTextureObject_t volTex2,
	float* proj,
	float3 s,
	const float3* __restrict__ cossinZT,
	const float* __restrict__ xds,
	const float* __restrict__ yds,
	const float* __restrict__ zds,
	const float* __restrict__ bxds,
	const float* __restrict__ byds,
	const float* __restrict__ bzds,
	float3 objCntIdx,
	float dx, float dz,
	int XN, int YN, int ZN,
	int DNU, int DNV, int PN)
{
	int detIdV = threadIdx.x + blockIdx.x * blockDim.x;
	int detIdU = threadIdx.y + blockIdx.y * blockDim.y;
	int angIdx = threadIdx.z + blockIdx.z * blockDim.z;
	__shared__ float _xds[BLKY];
	__shared__ float _yds[BLKY];
	_xds[threadIdx.y] = xds[detIdU];
	_yds[threadIdx.y] = yds[detIdU];
	__syncthreads();
	if (detIdU < DNU && detIdV < DNV && angIdx < PN)
	{
		float3 dir = cossinZT[angIdx];
		float3 cursour = make_float3(
			s.x * dir.x - s.y * dir.y,
			s.x * dir.y + s.y * dir.x,
			s.z + dir.z);
		s = cossinZT[angIdx];
		float summ = _xds[threadIdx.y] * s.x - _yds[threadIdx.y] * s.y;
		float obj = _xds[threadIdx.y] * s.y + _yds[threadIdx.y] * s.x;
		float realL = bxds[detIdU];
		float realR = byds[detIdU];
		float realU = bxds[detIdU + 1];
		float realD = byds[detIdU + 1];

		float2 curDetL = make_float2(
			realL * s.x - realR * s.y,
			realL * s.y + realR * s.x);
		float2 curDetR = make_float2(
			realU * s.x - realD * s.y,
			realU * s.y + realD * s.x);

		float4 curDet = make_float4(
			summ, obj, bzds[detIdV] + s.z,
			bzds[detIdV + 1] + s.z);

		dir = normalize(make_float3(summ, obj,
			zds[detIdV] + s.z) - cursour);

		summ = 0;
		obj = 0;
		float intersectLength, intersectHeight;
		float invdz = 1.0 / dz;
		float invdx = 1.0 / dx;

		float factL(1.0f);
		float factR(1.0f);
		float factU(1.0f);
		float factD(1.0f);

		float constVal = 0;
		if (fabsf(s.x) <= fabsf(s.y))
		{
			summ = 0;
			factL = (curDetL.y - cursour.y) / (curDetL.x - cursour.x);
			factR = (curDetR.y - cursour.y) / (curDetR.x - cursour.x);
			factU = (curDet.w - cursour.z) / (curDet.x - cursour.x);
			factD = (curDet.z - cursour.z) / (curDet.x - cursour.x);

			constVal = dx * dx * dz / fabsf(dir.x);
#pragma unroll
			for (int ii = 0; ii < XN; ++ii)
			{
				obj = (ii - objCntIdx.x) * dx;

				realL = (obj - curDetL.x) * factL + curDetL.y;
				realR = (obj - curDetR.x) * factR + curDetR.y;
				realU = (obj - curDet.x) * factU + curDet.w;
				realD = (obj - curDet.x) * factD + curDet.z;

				intersectLength = realR - realL;
				intersectHeight = realU - realD;


				realD = realD * invdz + objCntIdx.z + 1;
				realR = realR * invdx + objCntIdx.y + 1;
				realU = realU * invdz + objCntIdx.z + 1;
				realL = realL * invdx + objCntIdx.y + 1;

				summ +=
					(
					tex3D<float>(volTex2, realD, realL, ii + 0.5)
					+ tex3D<float>(volTex2, realU, realR, ii + 0.5)
					- tex3D<float>(volTex2, realU, realL, ii + 0.5)
					- tex3D<float>(volTex2, realD, realR, ii + 0.5)
					) / (intersectLength * intersectHeight);

			}
			__syncthreads();
			proj[(angIdx * DNU + detIdU) * DNV + detIdV] = summ * constVal;
		}
		else
		{
			summ = 0;
			factL = (curDetL.x - cursour.x) / (curDetL.y - cursour.y);
			factR = (curDetR.x - cursour.x) / (curDetR.y - cursour.y);
			factU = (curDet.w - cursour.z) / (curDet.y - cursour.y);
			factD = (curDet.z - cursour.z) / (curDet.y - cursour.y);

			constVal = dx * dx * dz / fabsf(dir.y);
#pragma unroll
			for (int jj = 0; jj < YN; ++jj)
			{
				obj = (jj - objCntIdx.y) * dx;

				realL = (obj - curDetL.y) * factL + curDetL.x;
				realR = (obj - curDetR.y) * factR + curDetR.x;
				realU = (obj - curDet.y) * factU + curDet.w;
				realD = (obj - curDet.y) * factD + curDet.z;

				intersectLength = realR - realL;
				intersectHeight = realU - realD;


				realD = realD * invdz + objCntIdx.z + 1;
				realR = realR * invdx + objCntIdx.x + 1;
				realU = realU * invdz + objCntIdx.z + 1;
				realL = realL * invdx + objCntIdx.x + 1;

				summ +=
					(
					tex3D<float>(volTex1, realD, realL, jj + 0.5)
					+ tex3D<float>(volTex1, realU, realR, jj + 0.5)
					- tex3D<float>(volTex1, realU, realL, jj + 0.5)
					- tex3D<float>(volTex1, realD, realR, jj + 0.5)
					) / (intersectLength * intersectHeight);

			}
			__syncthreads();
			proj[(angIdx * DNU + detIdU) * DNV + detIdV] = summ * constVal;
		}
	}
}








__global__ void DD3_gpu_proj_pseudodistancedriven_ker(
	cudaTextureObject_t volTex,
	float* proj, float3 s,
	float* d_xds, float* d_yds, float* d_zds,
	float3* cossinT,
	float3 objCntIdx,
	float dx, float dz,
	int XN, int YN,
	int DNU, int DNV, int PN)
{
	int detIdV = threadIdx.x + blockIdx.x * blockDim.x;
	int detIdU = threadIdx.y + blockIdx.y * blockDim.y;
	int angIdx = threadIdx.z + blockIdx.z * blockDim.z;
	if (detIdV < DNV && detIdU < DNU && angIdx < PN)
	{
		float3 cossin = cossinT[angIdx];
		float3 cursour = make_float3(
			s.x * cossin.x - s.y * cossin.y,
			s.x * cossin.y + s.y * cossin.x,
			s.z + cossin.z);

		float summ = d_xds[detIdU];
		float obj = d_yds[detIdU];
		float idx = d_zds[detIdV];
		float3 curDet = make_float3(
			summ * cossin.x - obj * cossin.y,
			summ * cossin.y + obj * cossin.x,
			idx + cossin.z);

		float3 dir = normalize(curDet - cursour);
		summ = 0;
		obj = 0;
		float idxZ;
		if (fabsf(cossin.x) <= fabsf(cossin.y))
		{
			summ = 0;
			for (int ii = 0; ii < XN; ++ii)
			{
				obj = (ii - objCntIdx.x) * dx;
				idx = (obj - curDet.x) / dir.x * dir.y + curDet.y;
				idxZ = (obj - curDet.x) / dir.x * dir.z + curDet.z;

				idx = idx / dx + objCntIdx.y + 0.5;
				idxZ = idxZ / dz + objCntIdx.z + 0.5;
				summ += tex3D<float>(volTex, idxZ, ii + 0.5f, idx);
			}
			__syncthreads();
			proj[(angIdx * DNU + detIdU) * DNV + detIdV] = summ * dx / fabsf(dir.x);
		}
		else
		{
			summ = 0;
			for (int jj = 0; jj != YN; ++jj)
			{
				obj = (jj - objCntIdx.y) * dx;
				idx = (obj - curDet.y) / dir.y * dir.x + curDet.x;
				idxZ = (obj - curDet.y) / dir.y * dir.z + curDet.z;

				idx = idx / dx + objCntIdx.x + 0.5;
				idxZ = idxZ / dz + objCntIdx.z + 0.5;
				summ += tex3D<float>(volTex, idxZ, idx, jj + 0.5f);
			}
			__syncthreads();
			proj[(angIdx * DNU + detIdU) * DNV + detIdV] = summ * dx / fabsf(dir.y);
		}
	}
}





////Use the split-collect method to do the projection
//void DD3ProjHelical_3GPU(
//	float x0, float y0, float z0,
//	int DNU, int DNV,
//	float* xds, float* yds, float* zds,
//	float imgXCenter, float imgYCenter, float imgZCenter,
//	float* hangs, float* hzPos, int PN,
//	int XN, int YN, int ZN,
//	float* hvol, float* hprj,
//	float dx, float dz,
//	byte* mask, int methodId, int (&startPN)[3])
//{
//
//}

// Divide three sub volumes.
template<typename T>
void GenSubVols(
		int* ObjIdx_Start,
		int* ObjIdx_End,
		int* SZN,
		T** subVol,
		T* subImgZCenter,
		const int subVolN,
		const int* PrjIdx_Start,
		const int* PrjIdx_End,
		const T detCntIdxV,
		const T detStpZ,
		const T objCntIdxZ,
		const T dz,
		const int ZN,
		const int DNV,
		const T imgZCenter,
		const int PN,
		const int XN,
		const int YN,
		const T* hvol,
		const T* hzPos)
{
	if (nullptr == ObjIdx_Start)
	{
		ObjIdx_Start = new int[subVolN];
	}
	if(nullptr == ObjIdx_End)
	{
		ObjIdx_End = new int[subVolN];
	}
	if(nullptr == SZN)
	{
		SZN = new int[subVolN];
	}
	if(nullptr == subVol)
	{
		subVol = new float*[subVolN];
	}
	if(nullptr == subImgZCenter)
	{
		subImgZCenter = new float[subVolN];
	}


	thrust::host_vector<T> h_zPos(hzPos, hzPos + PN);
	omp_set_num_threads(subVolN);
#pragma omp parallel for
	for(int i = 0; i < subVolN; ++i)  //The last one has problem!!!!!!!!!!
	{
		getVolZIdxPair<T>(h_zPos,PrjIdx_Start[i],PrjIdx_End[i],
					detCntIdxV, detStpZ, DNV, objCntIdxZ, dz, ZN, ObjIdx_Start[i], ObjIdx_End[i]);
		std::cout<<i<<" "<<ObjIdx_Start[i]<<" "<<ObjIdx_End[i]<<"\n";
		SZN[i] = ObjIdx_End[i] - ObjIdx_Start[i];
		subVol[i] = new T[XN * YN * SZN[i]];
		//Divide the volume
		getSubVolume<T>(hvol, XN * YN, ZN, ObjIdx_Start[i], ObjIdx_End[i], subVol[i]);
		//Calculate the corresponding center position
		subImgZCenter[i] = ((ObjIdx_End[i] + ObjIdx_Start[i] - (ZN - 1.0)) * dz + imgZCenter * 2.0) / 2.0;
	}
}

template<typename T>
void DelSubVols(
		int* ObjIdx_Start,
		int* ObjIdx_End,
		int* SZN,
		T** subVol,
		T* subImgZCenter, const int subVolN)
{
	for(int i = 0; i != subVolN; ++i)
	{
		delete[] subVol[i];
	}
	delete[] subVol;
	delete[] subImgZCenter;
	delete[] ObjIdx_Start;
	delete[] ObjIdx_End;
	delete[] SZN;
}


void DD3_gpu_proj_pseudodistancedriven_multiGPU(
		float x0, float y0, float z0,
		int DNU, int DNV,
		float* xds, float* yds, float* zds,
		float imgXCenter, float imgYCenter, float imgZCenter,
		float* h_angs, float* h_zPos, int PN,
		int XN, int YN, int ZN,
		float* hvol, float* hprj,
		float dx, float dz,
		byte* mask,const int* startPN, int gpuNum)
{
	thrust::host_vector<float> hangs(h_angs, h_angs + PN);
	thrust::host_vector<float> hzPos(h_zPos, h_zPos + PN);
	// Mask the volume
	for (int i = 0; i != XN * YN; ++i)
	{
		byte v = mask[i];
		for (int z = 0; z != ZN; ++z)
		{
			hvol[i * ZN + z] = hvol[i * ZN + z] * v;
		}
	}
	// Calculate the boundary positions

	const float objCntIdxX = (XN - 1.0) * 0.5 - imgXCenter / dx;
	const float objCntIdxY = (YN - 1.0) * 0.5 - imgYCenter / dx;
	const float objCntIdxZ = (ZN - 1.0) * 0.5 - imgZCenter / dz;

	// Divide the volume into sub volumes with overlaps according to the startPN
	std::vector<int> ObjIdx_Start(gpuNum, -1);
	std::vector<int> ObjIdx_End(gpuNum, -1);

	std::vector<int> PrjIdx_Start(startPN, startPN+gpuNum);
	std::vector<int> PrjIdx_End(gpuNum, 0);

	std::copy(PrjIdx_Start.begin()+1, PrjIdx_Start.end(), PrjIdx_End.begin());
	PrjIdx_End[gpuNum - 1] = PN;
	std::vector<int> SPN = PrjIdx_End - PrjIdx_Start;
	std::vector<int> prefixSPN = SPN;

	thrust::exclusive_scan(prefixSPN.begin(), prefixSPN.end(), prefixSPN.begin());
	//std::cout<<"prefixSPN are "<<prefixSPN[0]<<"  "<<prefixSPN[1]<<"  "<<prefixSPN[2]<<"\n";

	std::vector<int> SZN(gpuNum, 0); // The slices number of each sub volume
	const float detStpZ = (zds[DNV-1] - zds[0]) / (DNV - 1); // detector cell height
	const float detCntIdxV = -zds[0] / detStpZ; // Detector center along Z direction

	std::vector<std::vector<float> > subVol(gpuNum); // Used to store three sub volumes
	std::vector<float> subImgZCenter(gpuNum, 0); // the center of three sub volumes

	// Generate multiple streams;
	std::vector<cudaStream_t> stream(gpuNum);

	std::vector<int> siz(gpuNum, 0);
	std::vector<cudaExtent> volumeSize(gpuNum);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	std::vector<cudaArray*> d_volumeArray(gpuNum);

	thrust::host_vector<thrust::device_vector<float> > d_vol(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > prj(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > d_xds(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > d_yds(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > d_zds(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > angs(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > zPos(gpuNum);
	thrust::host_vector<thrust::device_vector<float3> > cossinZT(gpuNum);

	dim3 blk(64, 16, 1);
	std::vector<dim3> gid(gpuNum);
	std::vector<cudaTextureObject_t> texObj(gpuNum);
	// First we define the main framework to see how it works.
	omp_set_num_threads(gpuNum);
#pragma omp parallel for
	for(int i = 0; i < gpuNum; ++i)
	{
		getVolZIdxPair<float>(hzPos, PrjIdx_Start[i], PrjIdx_End[i],
				detCntIdxV, detStpZ, DNV, objCntIdxZ, dz, ZN, ObjIdx_Start[i],
				ObjIdx_End[i]);
		//std::cout<<i<<" "<<ObjIdx_Start[i]<<" "<<ObjIdx_End[i]<<"\n";
		SZN[i] = ObjIdx_End[i] - ObjIdx_Start[i];
		subVol[i].resize(XN * YN * SZN[i]);
		// Divide the volume into multiple sets
		getSubVolume<float>(hvol, XN * YN, ZN, ObjIdx_Start[i], ObjIdx_End[i], &(subVol[i][0]));

		// NOTE: The explanation will be later:
		subImgZCenter[i] = -imgZCenter / dz + ZN * 0.5 - ObjIdx_Start[i] - 0.5f;

		CUDA_SAFE_CALL(cudaSetDevice(i));
		// For each GPU generate two streams
		CUDA_SAFE_CALL(cudaStreamCreate(&stream[i]));
		siz[i] = XN * YN * SZN[i];

		d_vol[i].resize(siz[i]);
		d_vol[i] = subVol[i];
		subVol[i].clear();

		volumeSize[i].width = SZN[i];
		volumeSize[i].height = XN;
		volumeSize[i].depth = YN;
		CUDA_SAFE_CALL(cudaMalloc3DArray(&d_volumeArray[i], &channelDesc, volumeSize[i]));

		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr((void*)
			thrust::raw_pointer_cast(&d_vol[i][0]),
			volumeSize[i].width * sizeof(float),
			volumeSize[i].width, volumeSize[i].height);
		copyParams.dstArray = d_volumeArray[i];
		copyParams.extent = volumeSize[i];
		copyParams.kind = cudaMemcpyDeviceToDevice;

		CUDA_SAFE_CALL(cudaMemcpy3DAsync(&copyParams,stream[i]));
		d_vol[i].clear();


		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = d_volumeArray[i];

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeBorder;
		texDesc.addressMode[1] = cudaAddressModeBorder;
		texDesc.addressMode[2] = cudaAddressModeBorder;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = false;
		texObj[i] = 0;
		CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj[i], &resDesc, &texDesc, nullptr));


		prj[i].resize(DNU * DNV * SPN[i]); // Change here
		d_xds[i].resize(DNU);
		d_yds[i].resize(DNU);
		d_zds[i].resize(DNV);
		thrust::copy(xds,xds+DNU,d_xds[i].begin());
		thrust::copy(yds,yds+DNU,d_yds[i].begin());
		thrust::copy(zds,zds+DNV,d_zds[i].begin());

		angs[i].resize(SPN[i]);
		zPos[i].resize(SPN[i]);
		thrust::copy(hangs.begin() + PrjIdx_Start[i],
					 hangs.begin() + PrjIdx_Start[i] + SPN[i],
					 angs[i].begin());
		thrust::copy(hzPos.begin() + PrjIdx_Start[i],
					 hzPos.begin() + PrjIdx_Start[i] + SPN[i],
					 zPos[i].begin());
		cossinZT[i].resize(PN);

		thrust::transform(
			thrust::make_zip_iterator(thrust::make_tuple(angs[i].begin(), zPos[i].begin())),
			thrust::make_zip_iterator(thrust::make_tuple(angs[i].end(), zPos[i].end())),
			cossinZT[i].begin(), CTMBIR::ConstantForBackProjection4(x0, y0, z0));
		angs[i].clear();
		zPos[i].clear();

		gid[i].x = (DNV + blk.x - 1) / blk.x;
		gid[i].y = (DNU + blk.y - 1) / blk.y;
		gid[i].z = (SPN[i] + blk.z - 1) / blk.z;
	}
#pragma omp parallel for
	for(int i = 0; i < gpuNum; ++i)
	{
		cudaSetDevice(i);
		DD3_gpu_proj_pseudodistancedriven_ker<< <gid[i], blk, 0, stream[i]>> >(
				texObj[i], thrust::raw_pointer_cast(&prj[i][0]),
			make_float3(x0, y0, z0),
			thrust::raw_pointer_cast(&d_xds[i][0]),
			thrust::raw_pointer_cast(&d_yds[i][0]),
			thrust::raw_pointer_cast(&d_zds[i][0]),
			thrust::raw_pointer_cast(&cossinZT[i][0]),
			make_float3(objCntIdxX, objCntIdxY, subImgZCenter[i]),
			dx, dz, XN, YN, DNU, DNV, SPN[i]);
	}
#pragma omp barrier
#pragma omp parallel for
	for(int i = 0; i < gpuNum; ++i)
	{
		cudaSetDevice(i);
		CUDA_SAFE_CALL(cudaMemcpyAsync(hprj + DNU * DNV * prefixSPN[i],
				thrust::raw_pointer_cast(&prj[i][0]), sizeof(float) * DNU * DNV * SPN[i],
				cudaMemcpyDeviceToHost,stream[i]));
		d_xds[i].clear();
		d_yds[i].clear();
		d_zds[i].clear();
		cossinZT[i].clear();
		prj[i].clear();

		CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj[i]));
		CUDA_SAFE_CALL(cudaFreeArray(d_volumeArray[i]));
		//CUDA_SAFE_CALL(cudaStreamDestroy(stream[i*2]));
		//CUDA_SAFE_CALL(cudaStreamDestroy(stream[i*2 + 1]));
	}

	// Delete the vectors;
	hangs.clear();
	hzPos.clear();
	ObjIdx_Start.clear();
	ObjIdx_End.clear();
	PrjIdx_Start.clear();
	PrjIdx_End.clear();
	SPN.clear();
	prefixSPN.clear();
	SZN.clear();
	subVol.clear();
	subImgZCenter.clear();
	stream.clear();
	siz.clear();
	volumeSize.clear();
	d_volumeArray.clear();
	d_vol.clear();
	prj.clear();
	d_xds.clear();
	d_yds.clear();
	d_zds.clear();
	angs.clear();
	zPos.clear();
	cossinZT.clear();
	gid.clear();

}


void DD3_gpu_proj_branchless_sat2d_multiGPU(
		float x0, float y0, float z0,
		int DNU, int DNV,
		float* xds, float* yds, float* zds,
		float imgXCenter, float imgYCenter, float imgZCenter,
		float* h_angs, float* h_zPos, int PN,
		int XN, int YN, int ZN,
		float* hvol, float* hprj,
		float dx, float dz,
		byte* mask,const int* startPN, int gpuNum)
{
	thrust::host_vector<float> hangs(h_angs, h_angs+PN);
	thrust::host_vector<float> hzPos(h_zPos, h_zPos+PN);

	for (int i = 0; i != XN * YN; ++i)
	{
		byte v = mask[i];
		for (int z = 0; z != ZN; ++z)
		{
			hvol[i * ZN + z] = hvol[i * ZN + z] * v;
		}
	}
	// Calculate the boundary positions
	std::vector<float> bxds(DNU + 1, 0.0f);
	std::vector<float> byds(DNU + 1, 0.0f);
	std::vector<float> bzds(DNV + 1, 0.0f);

	DD3Boundaries<float>(DNU + 1, xds, bxds);
	DD3Boundaries<float>(DNU + 1, yds, byds);
	DD3Boundaries<float>(DNV + 1, zds, bzds);

	const float objCntIdxX = (XN - 1.0) * 0.5 - imgXCenter / dx;
	const float objCntIdxY = (YN - 1.0) * 0.5 - imgYCenter / dx;
	const float objCntIdxZ = (ZN - 1.0) * 0.5 - imgZCenter / dz;

	// Divide the volume into sub volumes with overlaps according to the startPN
	std::vector<int> ObjIdx_Start(gpuNum, -1);
	std::vector<int> ObjIdx_End(gpuNum, -1);

	std::vector<int> PrjIdx_Start(startPN, startPN+gpuNum);
	std::vector<int> PrjIdx_End(gpuNum, 0);

	std::copy(PrjIdx_Start.begin()+1, PrjIdx_Start.end(), PrjIdx_End.begin());
	PrjIdx_End[gpuNum - 1] = PN;
	std::vector<int> SPN = PrjIdx_End - PrjIdx_Start;
	std::vector<int> prefixSPN = SPN;
	thrust::exclusive_scan(prefixSPN.begin(), prefixSPN.end(), prefixSPN.begin());
	//std::cout<<"prefixSPN are "<<prefixSPN[0]<<"  "<<prefixSPN[1]<<"  "<<prefixSPN[2]<<"\n";

	std::vector<int> SZN(gpuNum, 0); // The slices number of each sub volume
	const float detStpZ = (zds[DNV-1] - zds[0]) / (DNV - 1); // detector cell height
	const float detCntIdxV = -zds[0] / detStpZ; // Detector center along Z direction

	std::vector<std::vector<float> > subVol(gpuNum); // Used to store three sub volumes
	std::vector<float> subImgZCenter(gpuNum, 0); // the center of three sub volumes

	// Generate multiple streams;
	std::vector<cudaStream_t> stream(gpuNum * 2);


	std::vector<int> siz(gpuNum, 0);
	std::vector<int> nsiz_ZXY(gpuNum, 0);
	std::vector<int> nsiz_ZYX(gpuNum, 0);
	std::vector<int> nZN(gpuNum,0);

	const int nXN = XN + 1;
	const int nYN = YN + 1;

	thrust::host_vector<thrust::device_vector<float> > d_vol(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > d_ZXY(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > d_ZYX(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > prj(gpuNum); // Change here
	thrust::host_vector<thrust::device_vector<float> > d_xds(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > d_yds(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > d_zds(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > d_bxds(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > d_byds(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > d_bzds(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > angs(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > zPos(gpuNum);
	thrust::host_vector<thrust::device_vector<float3> > cossinZT(gpuNum);

	// Copy to three volumes
	dim3 copyblk(64, 16, 1);
	std::vector<dim3> copygid(gpuNum);
	dim3 satblk1(32,1,1);
	dim3 satblk2(64,16,1);
	dim3 satgid1_1((nXN * YN + satblk1.x - 1) / satblk1.x, 1, 1);
	dim3 satgid1_2((nYN * XN + satblk1.x - 1) / satblk1.x, 1, 1);
	std::vector<dim3> satgid2_1(gpuNum);
	std::vector<dim3> satgid2_2(gpuNum);

	dim3 blk(BLKX, BLKY, BLKZ);
	std::vector<dim3> gid(gpuNum);

	std::vector<cudaExtent> volumeSize1(gpuNum);
	std::vector<cudaExtent> volumeSize2(gpuNum);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	std::vector<cudaArray*> d_volumeArray1(gpuNum);
	std::vector<cudaArray*> d_volumeArray2(gpuNum);

	std::vector<cudaTextureObject_t> texObj1(gpuNum);
	std::vector<cudaTextureObject_t> texObj2(gpuNum);

	omp_set_num_threads(gpuNum);
#pragma omp parallel for
	for(int i = 0; i < gpuNum; ++i)
	{
		getVolZIdxPair<float>(hzPos, PrjIdx_Start[i], PrjIdx_End[i],
				detCntIdxV, detStpZ, DNV, objCntIdxZ, dz, ZN, ObjIdx_Start[i],
				ObjIdx_End[i]);
		SZN[i] = ObjIdx_End[i] - ObjIdx_Start[i];
		subVol[i].resize(XN * YN * SZN[i]);
		// Divide the volume into multiple sets
		getSubVolume<float>(hvol, XN * YN, ZN, ObjIdx_Start[i], ObjIdx_End[i], &(subVol[i][0]));

		// NOTE: How it comes
		// We need to calculate the (ii - subImgZCenter[i]) * dz to define the
		// real physical position of the voxel.
		// Assume that the physical center of the whole volume is imgZCenter
		// The minimum lower position of the volume is imgZCenter - dz * N / 2;
		// Then the corresponding physical lower boundary position of ObjIdx_Start[i]
		// is --> imgZCenter - dz * N / 2 + ObjIdx_Start[i] * dz
		// while the corresponding physical center position of layer ObjIdx_Start[i]
		// is -->  imgZCenter - dz * N / 2 + ObjIdx_Start[i] * dz + 0.5 * dz
		// We need when ii==0 --> (ii - subImgZCenter[i]) * dz = imgZCenter - dz * N / 2 + ObjIdx_Start[i] * dz + 0.5 * dz
		// It means subImgZCenter[i] = -imgZCenter / dz + N / 2 - ObjIdx_Start[i] - 0.5;
		subImgZCenter[i] = -imgZCenter / dz + ZN * 0.5 - ObjIdx_Start[i] - 0.5f;


		CUDA_SAFE_CALL(cudaSetDevice(i));
		// For each GPU generate two streams
		CUDA_SAFE_CALL(cudaStreamCreate(&stream[i * 2]));
		CUDA_SAFE_CALL(cudaStreamCreate(&stream[i * 2 + 1]));
		siz[i] = XN * YN * SZN[i];
		nZN[i] = SZN[i] + 1;
		nsiz_ZXY[i] = nZN[i] * nXN * YN;
		nsiz_ZYX[i] = nZN[i] * nYN * XN;

		d_ZXY[i].resize(nsiz_ZXY[i]);
		d_ZYX[i].resize(nsiz_ZYX[i]);
		d_vol[i].resize(siz[i]);
		d_vol[i] = subVol[i];
		subVol[i].clear();

		copygid[i].x = (SZN[i] + copyblk.x - 1) / copyblk.x;
		copygid[i].y = (XN + copyblk.y - 1) / copyblk.y;
		copygid[i].z = (YN + copyblk.z - 1) / copyblk.z;
		naive_copyToTwoVolumes << <copygid[i], copyblk, 0, stream[2 * i] >> >(
				thrust::raw_pointer_cast(&d_vol[i][0]),
				thrust::raw_pointer_cast(&d_ZXY[i][0]),
				thrust::raw_pointer_cast(&d_ZYX[i][0]),
				XN,YN,SZN[i]);
		CUDA_SAFE_CALL(cudaStreamSynchronize(stream[2 * i]));
		CUDA_SAFE_CALL(cudaStreamSynchronize(stream[2 * i + 1]));

		d_vol[i].clear();
		// Generate the SAT for two volumes
		satgid2_1[i].x = (nZN[i] + satblk2.x - 1) / satblk2.x;
		satgid2_1[i].y = (YN + satblk2.y - 1) / satblk2.y;
		satgid2_1[i].z = 1;

		satgid2_2[i].x = (nZN[i] + satblk2.x - 1) / satblk2.x;
		satgid2_2[i].y = (XN + satblk2.y - 1) / satblk2.y;
		satgid2_2[i].z = 1;

		verticalIntegral << <satgid1_1, satblk1, 0, stream[2 * i] >> >(
				thrust::raw_pointer_cast(&d_ZXY[i][0]), nZN[i], nXN * YN);
		horizontalIntegral << <satgid2_1[i], satblk2, 0, stream[2 * i] >> >(
				thrust::raw_pointer_cast(&d_ZXY[i][0]), nXN, nZN[i], YN);
		verticalIntegral << <satgid1_2, satblk1, 0, stream[2 * i + 1] >> >(
				thrust::raw_pointer_cast(&d_ZYX[i][0]), nZN[i], nYN * XN);
		horizontalIntegral << <satgid2_2[i], satblk2, 0, stream[2 * i + 1] >> >(
				thrust::raw_pointer_cast(&d_ZYX[i][0]), nYN, nZN[i], XN);

		//Bind to the texture;
		volumeSize1[i].width = nZN[i];
		volumeSize1[i].height = nXN;
		volumeSize1[i].depth = YN;

		volumeSize2[i].width = nZN[i];
		volumeSize2[i].height = nYN;
		volumeSize2[i].depth = XN;

		CUDA_SAFE_CALL(cudaMalloc3DArray(&d_volumeArray1[i], &channelDesc, volumeSize1[i]));
		CUDA_SAFE_CALL(cudaMalloc3DArray(&d_volumeArray2[i], &channelDesc, volumeSize2[i]));

		cudaMemcpy3DParms copyParams1 = { 0 };
		copyParams1.srcPtr = make_cudaPitchedPtr((void*)
			thrust::raw_pointer_cast(&d_ZXY[i][0]),
			volumeSize1[i].width * sizeof(float),
			volumeSize1[i].width, volumeSize1[i].height);
		copyParams1.dstArray = d_volumeArray1[i];
		copyParams1.extent = volumeSize1[i];
		copyParams1.kind = cudaMemcpyDeviceToDevice;

		cudaMemcpy3DParms copyParams2 = { 0 };
		copyParams2.srcPtr = make_cudaPitchedPtr((void*)
			thrust::raw_pointer_cast(&d_ZYX[i][0]),
			volumeSize2[i].width * sizeof(float),
			volumeSize2[i].width, volumeSize2[i].height);
		copyParams2.dstArray = d_volumeArray2[i];
		copyParams2.extent = volumeSize2[i];
		copyParams2.kind = cudaMemcpyDeviceToDevice;

		CUDA_SAFE_CALL(cudaMemcpy3DAsync(&copyParams1,stream[2 * i]));
		CUDA_SAFE_CALL(cudaMemcpy3DAsync(&copyParams2,stream[2 * i + 1]));

		d_ZXY[i].clear();
		d_ZYX[i].clear();

		cudaResourceDesc resDesc1;
		cudaResourceDesc resDesc2;
		memset(&resDesc1, 0, sizeof(resDesc1));
		memset(&resDesc2, 0, sizeof(resDesc2));
		resDesc1.resType = cudaResourceTypeArray;
		resDesc2.resType = cudaResourceTypeArray;
		resDesc1.res.array.array = d_volumeArray1[i];
		resDesc2.res.array.array = d_volumeArray2[i];
		cudaTextureDesc texDesc1;
		cudaTextureDesc texDesc2;
		memset(&texDesc1, 0, sizeof(texDesc1));
		memset(&texDesc2, 0, sizeof(texDesc2));
		texDesc1.addressMode[0] = cudaAddressModeClamp;
		texDesc1.addressMode[1] = cudaAddressModeClamp;
		texDesc1.addressMode[2] = cudaAddressModeClamp;
		texDesc2.addressMode[0] = cudaAddressModeClamp;
		texDesc2.addressMode[1] = cudaAddressModeClamp;
		texDesc2.addressMode[2] = cudaAddressModeClamp;
		texDesc1.filterMode = cudaFilterModeLinear;
		texDesc2.filterMode = cudaFilterModeLinear;
		texDesc1.readMode = cudaReadModeElementType;
		texDesc2.readMode = cudaReadModeElementType;
		texDesc1.normalizedCoords = false;
		texDesc2.normalizedCoords = false;
		texObj1[i] = 0;
		texObj2[i] = 0;
		CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj1[i], &resDesc1, &texDesc1, nullptr));
		CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj2[i], &resDesc2, &texDesc2, nullptr));


		prj[i].resize(DNU * DNV * SPN[i]); // Change here
		d_xds[i].resize(DNU);
		d_yds[i].resize(DNU);
		d_zds[i].resize(DNV);
		thrust::copy(xds,xds+DNU,d_xds[i].begin());
		thrust::copy(yds,yds+DNU,d_yds[i].begin());
		thrust::copy(zds,zds+DNV,d_zds[i].begin());
		d_bxds[i].resize(bxds.size());
		d_bxds[i] = bxds;
		d_byds[i].resize(byds.size());
		d_byds[i] = byds;
		d_bzds[i].resize(bzds.size());
		d_bzds[i] = bzds;

		angs[i].resize(SPN[i]);
		zPos[i].resize(SPN[i]);
		thrust::copy(hangs.begin() + PrjIdx_Start[i],
					 hangs.begin() + PrjIdx_Start[i] + SPN[i],
					 angs[i].begin());
		thrust::copy(hzPos.begin() + PrjIdx_Start[i],
					 hzPos.begin() + PrjIdx_Start[i] + SPN[i],
					 zPos[i].begin());
		cossinZT[i].resize(PN);

		thrust::transform(
			thrust::make_zip_iterator(thrust::make_tuple(angs[i].begin(), zPos[i].begin())),
			thrust::make_zip_iterator(thrust::make_tuple(angs[i].end(), zPos[i].end())),
			cossinZT[i].begin(), CTMBIR::ConstantForBackProjection4(x0, y0, z0));
		angs[i].clear();
		zPos[i].clear();

		gid[i].x = (DNV + blk.x - 1) / blk.x;
		gid[i].y = (DNU + blk.y - 1) / blk.y;
		gid[i].z = (SPN[i] + blk.z - 1) / blk.z;

	}
#pragma omp parallel for
	for(int i = 0; i < gpuNum; ++i)
	{
		cudaSetDevice(i);
		DD3_gpu_proj_branchless_sat2d_ker << <gid[i], blk, 0, stream[i * 2]>> >(
				texObj1[i], texObj2[i],
			thrust::raw_pointer_cast(&prj[i][0]),
			make_float3(x0, y0, z0),
			thrust::raw_pointer_cast(&cossinZT[i][0]),
			thrust::raw_pointer_cast(&d_xds[i][0]),
			thrust::raw_pointer_cast(&d_yds[i][0]),
			thrust::raw_pointer_cast(&d_zds[i][0]),
			thrust::raw_pointer_cast(&d_bxds[i][0]),
			thrust::raw_pointer_cast(&d_byds[i][0]),
			thrust::raw_pointer_cast(&d_bzds[i][0]),
			make_float3(objCntIdxX, objCntIdxY, subImgZCenter[i]),
			dx, dz, XN, YN, ZN, DNU, DNV, SPN[i]);
	}
#pragma omp barrier
#pragma omp parallel for
	for(int i = 0; i < gpuNum; ++i)
	{
		cudaSetDevice(i);
		CUDA_SAFE_CALL(cudaMemcpyAsync(hprj + DNU * DNV * prefixSPN[i],
				thrust::raw_pointer_cast(&prj[i][0]), sizeof(float) * DNU * DNV * SPN[i],
				cudaMemcpyDeviceToHost,stream[2*i]));
		d_xds[i].clear();
		d_yds[i].clear();
		d_zds[i].clear();
		d_bxds[i].clear();
		d_byds[i].clear();
		d_bzds[i].clear();
		cossinZT[i].clear();
		prj[i].clear();

		CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj1[i]));
		CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj2[i]));
		CUDA_SAFE_CALL(cudaFreeArray(d_volumeArray1[i]));
		CUDA_SAFE_CALL(cudaFreeArray(d_volumeArray2[i]));
	}

	// Clear the vectors
	hangs.clear();
	hzPos.clear();
	bxds.clear();
	byds.clear();
	bzds.clear();
	ObjIdx_Start.clear();
	ObjIdx_End.clear();
	PrjIdx_Start.clear();
	PrjIdx_End.clear();
	SPN.clear();
	prefixSPN.clear();
	SZN.clear();
	subVol.clear();
	subImgZCenter.clear();
	stream.clear();
	siz.clear();
	nsiz_ZXY.clear();
	nsiz_ZYX.clear();
	nZN.clear();
	d_vol.clear();
	d_ZXY.clear();
	d_ZYX.clear();
	prj.clear();
	d_xds.clear();
	d_yds.clear();
	d_zds.clear();
	d_bxds.clear();
	d_byds.clear();
	d_bzds.clear();
	angs.clear();
	zPos.clear();
	cossinZT.clear();
	copygid.clear();
	satgid2_1.clear();
	satgid2_2.clear();
	gid.clear();
	volumeSize1.clear();
	volumeSize2.clear();
	d_volumeArray1.clear();
	d_volumeArray2.clear();

}



extern "C"
void DD3Proj_multiGPU(
	float x0, float y0, float z0,
	int DNU, int DNV,
	float* xds, float* yds, float* zds,
	float imgXCenter, float imgYCenter, float imgZCenter,
	float* hangs, float* hzPos, int PN,
	int XN, int YN, int ZN,
	float* hvol, float* hprj,
	float dx, float dz,
	byte* mask, int prjMode, const int* startPN, int gpuNum)
{
	switch(prjMode)
	{
	case 0: // Branchless DD model based multi-GPU projection
		DD3_gpu_proj_branchless_sat2d_multiGPU(x0, y0, z0, DNU, DNV,
				xds, yds, zds, imgXCenter, imgYCenter, imgZCenter,
				hangs, hzPos, PN, XN, YN, ZN, hvol, hprj, dx, dz,
				mask, startPN, gpuNum);
		break;
	default: // Pseudo DD based multi-GPUs projection
		DD3_gpu_proj_pseudodistancedriven_multiGPU(x0, y0, z0, DNU, DNV,
				xds, yds, zds, imgXCenter, imgYCenter, imgZCenter,
				hangs, hzPos, PN, XN, YN, ZN, hvol, hprj, dx, dz,
				mask, startPN, gpuNum);
		break;
	}
}















enum BackProjectionMethod{ _BRANCHLESS, _PSEUDODD, _ZLINEBRANCHLESS, _VOLUMERENDERING };

#ifndef CALDETPARAS
#define CALDETPARAS
float4 calDetParas(float* xds, float* yds, float* zds, float x0, float y0, float z0, int DNU, int DNV)
{
	float* bxds = new float[DNU + 1];
	float* byds = new float[DNU + 1];
	float* bzds = new float[DNV + 1];
	DD3Boundaries(DNU + 1, xds, bxds);
	DD3Boundaries(DNU + 1, yds, byds);
	DD3Boundaries(DNV + 1, zds, bzds);

	float ddv = (bzds[DNV] - bzds[0]) / DNV;
	float detCtrIdxV = (-(bzds[0] - z0) / ddv) - 0.5;
	float2 dir = normalize(make_float2(-x0, -y0));
	float2 dirL = normalize(make_float2(bxds[0] - x0, byds[0] - y0));
	float2 dirR = normalize(make_float2(bxds[DNU] - x0, byds[DNU] - y0));
	float dbeta = asin(dirL.x * dirR.y - dirL.y * dirR.x) / DNU;
	float minBeta = asin(dir.x * dirL.y - dir.y * dirL.x);
	float detCtrIdxU = -minBeta / dbeta - 0.5;
	delete [] bxds;
	delete [] byds;
	delete [] bzds;
	return make_float4(detCtrIdxU, detCtrIdxV, dbeta, ddv);

}


#endif

__global__ void addTwoSidedZeroBoarder(float* prjIn, float* prjOut,
	const int DNU, const int DNV, const int PN)
{
	int idv = threadIdx.x + blockIdx.x * blockDim.x;
	int idu = threadIdx.y + blockIdx.y * blockDim.y;
	int pn = threadIdx.z + blockIdx.z * blockDim.z;
	if (idu < DNU && idv < DNV && pn < PN)
	{
		int inIdx = (pn * DNU + idu) * DNV + idv;
		int outIdx = (pn * (DNU + 2) + (idu + 1)) * (DNV + 2) + idv + 1;
		prjOut[outIdx] = prjIn[inIdx];
	}
}


__global__ void addOneSidedZeroBoarder(const float* prj_in, float* prj_out, int DNU, int DNV, int PN)
{
	int idv = threadIdx.x + blockIdx.x * blockDim.x;
	int idu = threadIdx.y + blockIdx.y * blockDim.y;
	int pn = threadIdx.z + blockIdx.z * blockDim.z;
	if (idu < DNU && idv < DNV && pn < PN)
	{
		int i = (pn * DNU + idu) * DNV + idv;
		int ni = (pn * (DNU + 1) + (idu + 1)) * (DNV + 1) + idv + 1;
		prj_out[ni] = prj_in[i];
	}
}

__global__ void verticalIntegral2(float* prj, int ZN, int N)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N)
	{
		int currentHead = idx * ZN;
		for (int ii = 1; ii < ZN; ++ii)
		{
			prj[currentHead + ii] = prj[currentHead + ii] + prj[currentHead + ii - 1];
		}
	}
}



__global__ void heorizontalIntegral2(float* prj, int DNU, int DNV, int PN)
{
	int idv = threadIdx.x + blockIdx.x * blockDim.x;
	int pIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if (idv < DNV && pIdx < PN)
	{
		int headPrt = pIdx * DNU * DNV + idv;
		for (int ii = 1; ii < DNU; ++ii)
		{
			prj[headPrt + ii * DNV] = prj[headPrt + ii * DNV] + prj[headPrt + (ii - 1) * DNV];
		}
	}
}

thrust::device_vector<float> genSAT_of_Projection(
	float* hprj,
	int DNU, int DNV, int PN)
{
	const int siz = DNU * DNV * PN;
	const int nsiz = (DNU + 1) * (DNV + 1) * PN;
	thrust::device_vector<float> prjSAT(nsiz, 0);
	thrust::device_vector<float> prj(hprj, hprj + siz);
	dim3 copyBlk(64, 16, 1);
	dim3 copyGid(
		(DNV + copyBlk.x - 1) / copyBlk.x,
		(DNU + copyBlk.y - 1) / copyBlk.y,
		(PN + copyBlk.z - 1) / copyBlk.z);

	addOneSidedZeroBoarder << <copyGid, copyBlk >> >(
		thrust::raw_pointer_cast(&prj[0]),
		thrust::raw_pointer_cast(&prjSAT[0]),
		DNU, DNV, PN);
	const int nDNU = DNU + 1;
	const int nDNV = DNV + 1;

	copyBlk.x = 512;
	copyBlk.y = 1;
	copyBlk.z = 1;
	copyGid.x = (nDNU * PN + copyBlk.x - 1) / copyBlk.x;
	copyGid.y = 1;
	copyGid.z = 1;
	verticalIntegral2 << <copyGid, copyBlk >> >(
		thrust::raw_pointer_cast(&prjSAT[0]),
		nDNV, nDNU * PN);
	copyBlk.x = 64;
	copyBlk.y = 16;
	copyBlk.z = 1;
	copyGid.x = (nDNV + copyBlk.x - 1) / copyBlk.x;
	copyGid.y = (PN + copyBlk.y - 1) / copyBlk.y;
	copyGid.z = 1;


	heorizontalIntegral2 << <copyGid, copyBlk >> >(
		thrust::raw_pointer_cast(&prjSAT[0]),
		nDNU, nDNV, PN);

	return prjSAT;
}


void createTextureObject(
	cudaTextureObject_t& texObj,
	cudaArray* d_prjArray,
	int Width, int Height, int Depth,
	float* sourceData,
	cudaMemcpyKind memcpyKind,
	cudaTextureAddressMode addressMode,
	cudaTextureFilterMode textureFilterMode,
	cudaTextureReadMode textureReadMode,
	bool isNormalized)
{
	cudaExtent prjSize;
	prjSize.width = Width;
	prjSize.height = Height;
	prjSize.depth = Depth;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	cudaMalloc3DArray(&d_prjArray, &channelDesc, prjSize);
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(
		(void*) sourceData, prjSize.width * sizeof(float),
		prjSize.width, prjSize.height);
	copyParams.dstArray = d_prjArray;
	copyParams.extent = prjSize;
	copyParams.kind = memcpyKind;
	cudaMemcpy3D(&copyParams);
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = d_prjArray;
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = addressMode;
	texDesc.addressMode[1] = addressMode;
	texDesc.addressMode[2] = addressMode;
	texDesc.filterMode = textureFilterMode;
	texDesc.readMode = textureReadMode;
	texDesc.normalizedCoords = isNormalized;

	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
}



void destroyTextureObject(cudaTextureObject_t& texObj, cudaArray* d_array)
{
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(d_array);
}




template < BackProjectionMethod METHOD >
__global__ void DD3_gpu_back_ker(
	cudaTextureObject_t prjTexObj,
	float* vol,
	const byte* __restrict__ msk,
	const float3* __restrict__ cossinT,
	float3 s,
	float S2D,
	float3 curvox,
	float dx, float dz,
	float dbeta, float ddv,
	float2 detCntIdx,
	int3 VN,
	int PN, int squared)
{}

template<>
__global__ void DD3_gpu_back_ker<_BRANCHLESS>(
	cudaTextureObject_t prjTexObj,
	float* vol,
	const byte* __restrict__ msk,
	const float3* __restrict__ cossinT,
	float3 s,
	float S2D,
	float3 curvox,
	float dx, float dz,
	float dbeta, float ddv,
	float2 detCntIdx,
	int3 VN,
	int PN, int squared)
{
	int3 id;
	id.z = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
	id.x = threadIdx.y + __umul24(blockIdx.y, blockDim.y);
	id.y = threadIdx.z + __umul24(blockIdx.z, blockDim.z);
	if (id.x < VN.x && id.y < VN.y && id.z < VN.z)
	{
		if (msk[id.y * VN.x + id.x] != 1)
			return;
		curvox = (id - curvox) * make_float3(dx, dx, dz);
		float3 cursour;
		float idxL, idxR, idxU, idxD;
		float cosVal;
		float summ = 0;

		float3 cossin;
		float inv_sid = 1.0 / sqrtf(s.x * s.x + s.y * s.y);
		float3 dir;
		float l_square;
		float l;
		float alpha;
		float deltaAlpha;
		S2D = S2D / ddv;
		dbeta = 1.0 / dbeta;
		dz = dz * 0.5;
		for (int angIdx = 0; angIdx < PN; ++angIdx)
		{
			cossin = cossinT[angIdx];
			cursour = make_float3(
				s.x * cossin.x - s.y * cossin.y,
				s.x * cossin.y + s.y * cossin.x,
				s.z + cossin.z);

			dir = curvox - cursour;
			l_square = dir.x * dir.x + dir.y * dir.y;
			l = rsqrtf(l_square);
			idxU = (dir.z + dz) * S2D * l + detCntIdx.y + 1;
			idxD = (dir.z - dz) * S2D * l + detCntIdx.y + 1;

			alpha = asinf((cursour.y * dir.x - cursour.x * dir.y) * inv_sid * l);
			if (fabsf(cursour.x) > fabsf(cursour.y))
			{
				ddv = dir.x;
			}
			else
			{
				ddv = dir.y;
			}
			deltaAlpha = ddv / l_square * dx * 0.5;
			cosVal = dx / ddv * sqrtf(l_square + dir.z * dir.z);
			idxL = (alpha - deltaAlpha) * dbeta + detCntIdx.x + 1;
			idxR = (alpha + deltaAlpha) * dbeta + detCntIdx.x + 1;

			summ +=
				(-tex3D<float>(prjTexObj, idxD, idxR, angIdx + 0.5)
				- tex3D<float>(prjTexObj, idxU, idxL, angIdx + 0.5)
				+ tex3D<float>(prjTexObj, idxD, idxL, angIdx + 0.5)
				+ tex3D<float>(prjTexObj, idxU, idxR, angIdx + 0.5)) * cosVal;
		}
		__syncthreads();
		vol[__umul24((__umul24(id.y, VN.x) + id.x), VN.z) + id.z] = summ;
	}
}





template<>
__global__ void DD3_gpu_back_ker<_PSEUDODD>(
	cudaTextureObject_t texObj,
	float* vol,
	const byte* __restrict__ msk,
	const float3* __restrict__ cossinZT,
	float3 s,
	float S2D,
	float3 objCntIdx,
	float dx, float dz, float dbeta, float ddv,
	float2 detCntIdx,
	int3 VN, int PN, int squared)
{
	int k = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
	int j = __mul24(blockIdx.z, blockDim.z) + threadIdx.z;
	if (i < VN.x && j < VN.y && k < VN.z)
	{
		if (msk[j * VN.x + i] != 1)
			return;
		float3 curVox = make_float3(
			(i - objCntIdx.x) * dx,
			(j - objCntIdx.y) * dx,
			(k - objCntIdx.z) * dz);

		float3 dir;
		float3 cursour;
		float invsid = rsqrtf(s.x * s.x + s.y * s.y);
		float invl;
		float idxZ;
		float idxXY;
		float alpha;
		float cosVal;
		float3 cossinT;
		float summ = 0;
		S2D = S2D / ddv;
		dbeta = 1.0 / dbeta;
		for (int angIdx = 0; angIdx != PN; ++angIdx)
		{
			cossinT = cossinZT[angIdx];
			cursour = make_float3(
				s.x * cossinT.x - s.y * cossinT.y,
				s.x * cossinT.y + s.y * cossinT.x,
				s.z + cossinT.z);

			dir = curVox - cursour;
			ddv = dir.x * dir.x + dir.y * dir.y;
			invl = rsqrtf(ddv);
			idxZ = dir.z * S2D * invl + detCntIdx.y + 0.5;
			alpha = asinf((cursour.y * dir.x - cursour.x * dir.y) * invl * invsid);
			if (fabsf(cursour.x) >= fabsf(cursour.y))
			{
				cosVal = fabsf(1.0 / dir.x);
			}
			else
			{
				cosVal = fabsf(1.0 / dir.y);
			}
			cosVal *= (dx * sqrtf(ddv + dir.z * dir.z));
			idxXY = alpha * dbeta + detCntIdx.x + 0.5;
			summ += tex3D<float>(texObj, idxZ, idxXY, angIdx + 0.5f) * cosVal;
		}
		__syncthreads();
		vol[(j * VN.x + i) * VN.z + k] = summ;
	}
}













void DD3Back_branchless_sat2d_multiGPU(
	float x0, float y0, float z0,
	int DNU, int DNV,
	float* xds, float* yds, float* zds,
	float imgXCenter, float imgYCenter, float imgZCenter,
	float* h_angs, float* h_zPos, int PN,
	int XN, int YN, int ZN,
	float* hvol, float* hprj,
	float dx, float dz,
	byte* mask,const int* startVOL, int gpuNum)
{
	const int nDNU = DNU + 1;
	const int nDNV = DNV + 1;

	thrust::host_vector<float> hangs(h_angs, h_angs + PN);
	thrust::host_vector<float> hzPos(h_zPos, h_zPos + PN);

	std::vector<int> ObjZIdx_Start(startVOL, startVOL + gpuNum);
	std::vector<int> ObjZIdx_End(ObjZIdx_Start.size());
	std::copy(ObjZIdx_Start.begin() + 1, ObjZIdx_Start.end(), ObjZIdx_End.begin());
	ObjZIdx_End[gpuNum - 1] = ZN;

	std::vector<int> prjIdx_Start(gpuNum);
	std::vector<int> prjIdx_End(gpuNum);

	const float objCntIdxZ = (ZN - 1.0f) * 0.5 - imgZCenter / dz;
	const float detStpZ = (zds[DNV - 1] - zds[0]) / (DNV - 1.0f); // detector cell height
	const float detCntIdxV = -zds[0] / detStpZ; // Detector Center along Z direction

	std::vector<int> SZN = ObjZIdx_End - ObjZIdx_Start; // sub volume slices number

	std::vector<float> subImgZCenter(gpuNum,0.0f);
	std::vector<int> SPN(gpuNum);

	const float objCntIdxX = (XN - 1.0f) * 0.5f - imgXCenter / dx;
	const float objCntIdxY = (YN - 1.0f) * 0.5f - imgYCenter / dx;

	std::vector<float3> sour(gpuNum);
	thrust::host_vector<thrust::device_vector<byte> > msk(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > vol(gpuNum);
	thrust::host_vector<thrust::device_vector<float3> > cossinZT(gpuNum);
	thrust::host_vector<cudaArray*> d_prjArray(gpuNum);
	thrust::host_vector<cudaTextureObject_t> texObj(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > prjSAT(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > prj(gpuNum);
	thrust::host_vector<cudaStream_t> stream(gpuNum);

	const float4 detParas = calDetParas(xds, yds, zds, x0, y0, z0, DNU, DNV);
	const float S2D = hypotf(xds[0] - x0, yds[0] - y0);

	// Pre calculate the cossin z positions
	thrust::device_vector<float3> COSSINZT(PN);
	thrust::device_vector<float> ANGS = hangs;
	thrust::device_vector<float> ZPOS = hzPos;
	thrust::transform(
		thrust::make_zip_iterator(thrust::make_tuple(ANGS.begin(), ZPOS.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(ANGS.end(), ZPOS.end())),
		COSSINZT.begin(), CTMBIR::ConstantForBackProjection4(x0, y0, z0));

	dim3 copyBlk(64,16,1);
	thrust::host_vector<dim3> copyGid(gpuNum);
	dim3 blk(BACK_BLKX, BACK_BLKY, BACK_BLKZ);
	thrust::host_vector<dim3> gid(gpuNum);
	dim3 vertGenBlk(512,1,1);
	thrust::host_vector<dim3> vertGenGid(gpuNum);
	dim3 horzGenBlk(64,16,1);
	thrust::host_vector<dim3> horzGenGid(gpuNum);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	thrust::host_vector<thrust::host_vector<float> > subVol(gpuNum);

	std::vector<size_t> siz(gpuNum,0);
	std::vector<size_t> nsiz(gpuNum,0);

	omp_set_num_threads(gpuNum);
#pragma omp parallel for
	for(int i = 0; i < gpuNum; ++i)
	{
		// get projection view index pair
		getPrjIdxPair<float>(hzPos, ObjZIdx_Start[i], ObjZIdx_End[i],
						objCntIdxZ, dz, ZN, detCntIdxV, detStpZ, DNV,
						prjIdx_Start[i], prjIdx_End[i]);
		SPN[i] = prjIdx_End[i] - prjIdx_Start[i];
		//std::cout<<i<<" "<<prjIdx_Start[i]<<" "<<prjIdx_End[i]<<"\n";
		// Calculate the corresponding center position index of the sub volumes
		subImgZCenter[i] = -imgZCenter / dz + ZN * 0.5 - ObjZIdx_Start[i] - 0.5f; // index position

		cudaSetDevice(i);
		cudaStreamCreate(&stream[i]);

		// Generate the SAT for the projection data
		siz[i] = DNU * DNV * SPN[i];
		nsiz[i] = (DNU + 1) * (DNV + 1) * SPN[i];
		prjSAT[i].resize(nsiz[i]);
		prj[i].resize(siz[i]);
		thrust::copy(
				hprj + DNU * DNV * prjIdx_Start[i],
				hprj + DNU * DNV * prjIdx_End[i],
				prj[i].begin());

		copyGid[i].x = (DNV + copyBlk.x - 1) / copyBlk.x;
		copyGid[i].y = (DNU + copyBlk.y - 1) / copyBlk.y;
		copyGid[i].z = (SPN[i] + copyBlk.z - 1) / copyBlk.z;
		addOneSidedZeroBoarder<<<copyGid[i], copyBlk, 0, stream[i]>>>(
				thrust::raw_pointer_cast(&prj[i][0]),
				thrust::raw_pointer_cast(&prjSAT[i][0]),
				DNU, DNV, SPN[i]);
		//cudaStreamSynchronize(stream[i]);

		vertGenGid[i].x = (nDNU * SPN[i] + vertGenBlk.x - 1) / copyBlk.x;
		vertGenGid[i].y = 1;
		vertGenGid[i].z = 1;
		verticalIntegral2 << <vertGenGid[i], vertGenBlk, 0, stream[i] >> >(
			thrust::raw_pointer_cast(&prjSAT[i][0]), nDNV, nDNU * SPN[i]);

		horzGenGid[i].x = (nDNV + horzGenBlk.x - 1) / horzGenBlk.x;
		horzGenGid[i].y = (SPN[i] + horzGenBlk.y - 1) / horzGenBlk.y;
		horzGenGid[i].z = 1;
		heorizontalIntegral2 << <horzGenGid[i], horzGenBlk,0,stream[i] >> >(
			thrust::raw_pointer_cast(&prjSAT[i][0]), nDNU, nDNV, SPN[i]);

		prj[i].clear();

		cudaExtent prjSize;
		prjSize.width = DNV + 1;
		prjSize.height = DNU + 1;
		prjSize.depth = SPN[i];
		cudaMalloc3DArray(&d_prjArray[i], &channelDesc, prjSize);

		cudaMemcpy3DParms copyParams = { 0 };
			copyParams.srcPtr = make_cudaPitchedPtr(
				(void*) thrust::raw_pointer_cast(&prjSAT[i][0]),
				prjSize.width * sizeof(float),
				prjSize.width, prjSize.height);
			copyParams.dstArray = d_prjArray[i];
			copyParams.extent = prjSize;
			copyParams.kind = cudaMemcpyDeviceToDevice;
		cudaMemcpy3DAsync(&copyParams,stream[i]);

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = d_prjArray[i];
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.addressMode[2] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = false;
		cudaCreateTextureObject(&texObj[i], &resDesc, &texDesc, nullptr);
		prjSAT[i].clear();
		// The part above are for branchless DD

		gid[i].x = (SZN[i] + blk.x - 1) / blk.x;
		gid[i].y = (XN + blk.y - 1) / blk.y;
		gid[i].z = (YN + blk.z - 1) / blk.z;

		vol[i].resize(XN * YN * SZN[i]);
		msk[i].resize(XN * YN);
		thrust::copy(mask, mask + XN * YN, msk[i].begin());

		cossinZT[i].resize(SPN[i]);
		thrust::copy(
				COSSINZT.begin() + prjIdx_Start[i],
				COSSINZT.begin() + prjIdx_End[i],
				cossinZT[i].begin());
	}
#pragma omp parallel for
	for(int i = 0; i < gpuNum; ++i)
	{

		cudaSetDevice(i);
		DD3_gpu_back_ker<_BRANCHLESS> << <gid[i], blk, 0, stream[i] >> >(texObj[i],
				thrust::raw_pointer_cast(&vol[i][0]), thrust::raw_pointer_cast(&msk[i][0]),
				thrust::raw_pointer_cast(&cossinZT[i][0]), make_float3(x0, y0, z0), S2D,
				make_float3(objCntIdxX, objCntIdxY, subImgZCenter[i]), //  have to be changed
				dx, dz, detParas.z, detParas.w, make_float2(detParas.x, detParas.y),
				make_int3(XN, YN, SZN[i]), SPN[i], 0);
	}
#pragma omp barrier
#pragma omp parallel for
	for(int i = 0 ;i < gpuNum; ++i)
	{
		cudaSetDevice(i);
		// copy the volume back.
		subVol[i].resize(XN * YN * SZN[i]);
		thrust::copy(vol[i].begin(), vol[i].end(), subVol[i].begin());

		vol[i].clear();
		msk[i].clear();
		cossinZT[i].clear();

		cudaDestroyTextureObject(texObj[i]);
		cudaFreeArray(d_prjArray[i]);
	}
	cudaDeviceSynchronize();

	combineVolume<float>(hvol, XN, YN, ZN, subVol, &(SZN[0]), gpuNum);

	hangs.clear();
	hzPos.clear();
	ObjZIdx_Start.clear();
	ObjZIdx_End.clear();
	prjIdx_Start.clear();
	prjIdx_End.clear();
	SZN.clear();
	subImgZCenter.clear();
	SPN.clear();
	sour.clear();
	msk.clear();
	vol.clear();
	cossinZT.clear();
	d_prjArray.clear();
	texObj.clear();
	prjSAT.clear();
	prj.clear();
	stream.clear();
	COSSINZT.clear();
	ANGS.clear();
	ZPOS.clear();
	copyGid.clear();
	gid.clear();
	vertGenGid.clear();
	horzGenGid.clear();
}





void DD3Back_pseudo_multiGPU(
	float x0, float y0, float z0,
	int DNU, int DNV,
	float* xds, float* yds, float* zds,
	float imgXCenter, float imgYCenter, float imgZCenter,
	float* h_angs, float* h_zPos, int PN,
	int XN, int YN, int ZN,
	float* hvol, float* hprj,
	float dx, float dz,
	byte* mask,const int* startVOL, int gpuNum)
{
	thrust::host_vector<float> hangs(h_angs, h_angs + PN);
	thrust::host_vector<float> hzPos(h_zPos, h_zPos + PN);

	std::vector<int> ObjZIdx_Start(startVOL, startVOL + gpuNum);
	std::vector<int> ObjZIdx_End(ObjZIdx_Start.size());
	std::copy(ObjZIdx_Start.begin() + 1, ObjZIdx_Start.end(), ObjZIdx_End.begin());
	ObjZIdx_End[gpuNum - 1] = ZN;

	std::vector<int> prjIdx_Start(gpuNum);
	std::vector<int> prjIdx_End(gpuNum);

	const float objCntIdxZ = (ZN - 1.0f) * 0.5 - imgZCenter / dz;
	const float detStpZ = (zds[DNV - 1] - zds[0]) / (DNV - 1.0f); // detector cell height
	const float detCntIdxV = -zds[0] / detStpZ; // Detector Center along Z direction

	std::vector<int> SZN = ObjZIdx_End - ObjZIdx_Start; // sub volume slices number

	std::vector<float> subImgZCenter(gpuNum,0.0f);
	std::vector<int> SPN(gpuNum);

	const float objCntIdxX = (XN - 1.0f) * 0.5f - imgXCenter / dx;
	const float objCntIdxY = (YN - 1.0f) * 0.5f - imgYCenter / dx;

	std::vector<float3> sour(gpuNum);
	thrust::host_vector<thrust::device_vector<byte> > msk(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > vol(gpuNum);
	thrust::host_vector<thrust::device_vector<float3> > cossinZT(gpuNum);
	thrust::host_vector<cudaArray*> d_prjArray(gpuNum);
	thrust::host_vector<cudaTextureObject_t> texObj(gpuNum);
	thrust::host_vector<thrust::device_vector<float> > prj(gpuNum);
	thrust::host_vector<cudaStream_t> stream(gpuNum);

	const float4 detParas = calDetParas(xds, yds, zds, x0, y0, z0, DNU, DNV);
	const float S2D = hypotf(xds[0] - x0, yds[0] - y0);

	// Pre calculate the cossin z positions
	thrust::device_vector<float3> COSSINZT(PN);
	thrust::device_vector<float> ANGS = hangs;
	thrust::device_vector<float> ZPOS = hzPos;
	thrust::transform(
		thrust::make_zip_iterator(thrust::make_tuple(ANGS.begin(), ZPOS.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(ANGS.end(), ZPOS.end())),
		COSSINZT.begin(), CTMBIR::ConstantForBackProjection4(x0, y0, z0));

	dim3 copyBlk(64,16,1);
	thrust::host_vector<dim3> copyGid(gpuNum);
	dim3 blk(BACK_BLKX, BACK_BLKY, BACK_BLKZ);
	thrust::host_vector<dim3> gid(gpuNum);
	dim3 vertGenBlk(512,1,1);
	thrust::host_vector<dim3> vertGenGid(gpuNum);
	dim3 horzGenBlk(64,16,1);
	thrust::host_vector<dim3> horzGenGid(gpuNum);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	thrust::host_vector<thrust::host_vector<float> > subVol(gpuNum);

	std::vector<size_t> siz(gpuNum,0);
	std::vector<size_t> nsiz(gpuNum,0);

	omp_set_num_threads(gpuNum);
#pragma omp parallel for
	for(int i = 0; i < gpuNum; ++i)
	{
		// get projection view index pair
		getPrjIdxPair<float>(hzPos, ObjZIdx_Start[i], ObjZIdx_End[i],
						objCntIdxZ, dz, ZN, detCntIdxV, detStpZ, DNV,
						prjIdx_Start[i], prjIdx_End[i]);
		SPN[i] = prjIdx_End[i] - prjIdx_Start[i];
		//std::cout<<i<<" "<<prjIdx_Start[i]<<" "<<prjIdx_End[i]<<"\n";
		// Calculate the corresponding center position index of the sub volumes
		subImgZCenter[i] = -imgZCenter / dz + ZN * 0.5 - ObjZIdx_Start[i] - 0.5f; // index position

		cudaSetDevice(i);
		cudaStreamCreate(&stream[i]);
		////////////////////////////////////////////////////////////////////////
		siz[i] = DNU * DNV * SPN[i];
		prj[i].resize(siz[i]);
		thrust::copy(
				hprj + DNU * DNV * prjIdx_Start[i],
				hprj + DNU * DNV * prjIdx_End[i],
				prj[i].begin());

		cudaExtent prjSize;
		prjSize.width = DNV;
		prjSize.height = DNU;
		prjSize.depth = SPN[i];
		cudaMalloc3DArray(&d_prjArray[i], &channelDesc, prjSize);

		cudaMemcpy3DParms copyParams = { 0 };
			copyParams.srcPtr = make_cudaPitchedPtr(
				(void*) thrust::raw_pointer_cast(&prj[i][0]),
				prjSize.width * sizeof(float),
				prjSize.width, prjSize.height);
			copyParams.dstArray = d_prjArray[i];
			copyParams.extent = prjSize;
			copyParams.kind = cudaMemcpyDeviceToDevice;
		cudaMemcpy3DAsync(&copyParams,stream[i]);

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = d_prjArray[i];
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeBorder;
		texDesc.addressMode[1] = cudaAddressModeBorder;
		texDesc.addressMode[2] = cudaAddressModeBorder;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = false;
		cudaCreateTextureObject(&texObj[i], &resDesc, &texDesc, nullptr);
		prj[i].clear();
		////////////////////////////////////////////////////////////////////////
		// Generate the SAT for the projection data
		// The part above are for branchless DD

		gid[i].x = (SZN[i] + blk.x - 1) / blk.x;
		gid[i].y = (XN + blk.y - 1) / blk.y;
		gid[i].z = (YN + blk.z - 1) / blk.z;

		vol[i].resize(XN * YN * SZN[i]);
		msk[i].resize(XN * YN);
		thrust::copy(mask, mask + XN * YN, msk[i].begin());

		cossinZT[i].resize(SPN[i]);
		thrust::copy(
				COSSINZT.begin() + prjIdx_Start[i],
				COSSINZT.begin() + prjIdx_End[i],
				cossinZT[i].begin());
	}
#pragma omp parallel for
	for(int i = 0; i < gpuNum; ++i)
	{
		cudaSetDevice(i);
		DD3_gpu_back_ker<_PSEUDODD> << <gid[i], blk, 0, stream[i] >> >(texObj[i],
				thrust::raw_pointer_cast(&vol[i][0]), thrust::raw_pointer_cast(&msk[i][0]),
				thrust::raw_pointer_cast(&cossinZT[i][0]), make_float3(x0, y0, z0), S2D,
				make_float3(objCntIdxX, objCntIdxY, subImgZCenter[i]), //  have to be changed
				dx, dz, detParas.z, detParas.w, make_float2(detParas.x, detParas.y),
				make_int3(XN, YN, SZN[i]), SPN[i], 0);
	}
#pragma omp barrier
#pragma omp parallel for
	for (int i = 0; i < gpuNum; ++i)
	{
		// copy the volume back.
		subVol[i].resize(XN * YN * SZN[i]);
		thrust::copy(vol[i].begin(), vol[i].end(), subVol[i].begin());

		vol[i].clear();
		msk[i].clear();
		cossinZT[i].clear();

		cudaDestroyTextureObject(texObj[i]);
		cudaFreeArray(d_prjArray[i]);
	}
	cudaDeviceSynchronize();

	combineVolume<float>(hvol, XN, YN, ZN, subVol, &(SZN[0]), gpuNum);

	hangs.clear();
	hzPos.clear();
	ObjZIdx_Start.clear();
	ObjZIdx_End.clear();
	prjIdx_Start.clear();
	prjIdx_End.clear();
	SZN.clear();
	subImgZCenter.clear();
	SPN.clear();
	sour.clear();
	msk.clear();
	vol.clear();
	cossinZT.clear();
	d_prjArray.clear();
	texObj.clear();
	prj.clear();
	stream.clear();
	COSSINZT.clear();
	ANGS.clear();
	ZPOS.clear();
	copyGid.clear();
	gid.clear();
	vertGenGid.clear();
	horzGenGid.clear();
}




extern "C"
void DD3Back_multiGPU(
	float x0, float y0, float z0,
	int DNU, int DNV,
	float* xds, float* yds, float* zds,
	float imgXCenter, float imgYCenter, float imgZCenter,
	float* hangs, float* hzPos, int PN,
	int XN, int YN, int ZN,
	float* hvol, float* hprj,
	float dx, float dz,
	byte* mask, int bakMode,const int* startVOL, int gpuNum)
{
	switch(bakMode)
	{
	case 0: // Branchless backprojection
		DD3Back_branchless_sat2d_multiGPU(x0, y0, z0,
			DNU, DNV, xds, yds, zds, imgXCenter, imgYCenter, imgZCenter,
			hangs, hzPos, PN, XN, YN, ZN, hvol, hprj,
			dx, dz, mask, startVOL, gpuNum);
		break;
	default: // Volume Rendering backprojection
		DD3Back_pseudo_multiGPU(x0, y0, z0,
			DNU, DNV, xds, yds, zds, imgXCenter, imgYCenter, imgZCenter,
			hangs, hzPos, PN, XN, YN, ZN, hvol, hprj,
			dx, dz, mask, startVOL, gpuNum);
		break;
	}

}
