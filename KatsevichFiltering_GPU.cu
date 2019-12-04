/*
============================================================================
Name        : KatsevichFiltering_GPU.cu
Author      : Rui Liu
Version     : 1.0
Copyright   : Your copyright notice
Description : Compute the filtering of the projection data
============================================================================
*/

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <memory>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>

#include <fstream>

#include "device_launch_parameters.h"
#include <stdio.h>
#include <mex.h>

static const double PI = 3.14159265358979323846264;
#define TWOPI (2.0 * 3.14159265358)
#define DELTAMAX (1.0e-7)


template<typename T>
void writeVectorToDisk(const std::string& name, thrust::device_vector<T>& v, int x, int y, int z) {
    std::string nam = name + "_" + std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z) + ".data";
    std::ofstream fou(nam, std::ios::binary);
    thrust::host_vector<T> hostVec = v;

    fou.write((char*)(&(hostVec[0])), x * y * z * sizeof(T));
    fou.close();
}

// Katsevich Filtering 1. Chain rule 2. Direct method

// The main parameters are
// 1. Proj 				: 	projection data
// 2. ProjScale 		:  	?
// 3. DecWidth			: 	?
// 4. DecHeight			: 	?
// 5. ScanR				:	source to object distance
// 6. StdDis			: 	source to detector distance
// 7. HelicP			:	helical pitch
// 8. FilterCoeFF		: 	filter coefficient?
// 9. DeltaAngle		:	?
// 10.FilteringMode		: 	?


// The projection is arranged as (YL, ZL, ProjNumber in MATLAB)

// Function 1 Compute the derivative of projection
template<typename T>
__global__ void DevAlongProjView_ker(T* GF, const T* proj, const int YL, const int ZL, const int ProjNumber, const T invTWODeltaL) // invTWODelta = 1.0 / (2.0 * DeltaL)
{
	int yIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int zIdx = threadIdx.y + blockIdx.y * blockDim.y;
	int pIdx = threadIdx.z + blockIdx.z * blockDim.z;
	if (yIdx < YL && zIdx < ZL && pIdx < ProjNumber - 1 && pIdx > 0)
	{
		int lidx = ((pIdx - 1) * ZL + zIdx) * YL + yIdx;
		int ridx = ((pIdx + 1) * ZL + zIdx) * YL + yIdx;
		int idx = (pIdx * ZL + zIdx) * YL + yIdx;
		GF[idx] = (proj[ridx] - proj[lidx]) * invTWODeltaL;
	}
}

template<typename T>
void DevAlongProjview_gpu(thrust::device_vector<T>& GF, const thrust::device_vector<T>& Proj, const int YL, const int ZL, const int ProjNumber, const T DeltaL)
{
	const T invTWODeltaL = 1.0 / (2.0 * DeltaL);
	dim3 blk(64, 8, 1);
	dim3 gid((YL + blk.x - 1) / blk.x,
		(ZL + blk.y - 1) / blk.y,
		(ProjNumber + blk.z - 1) / blk.z);
	DevAlongProjView_ker<T> << <gid, blk >> >(
		thrust::raw_pointer_cast(&GF[0]),
		thrust::raw_pointer_cast(&Proj[0]),
		YL, ZL, ProjNumber, invTWODeltaL);

	//Dealing with the edging case
	if (ProjNumber >= 2) // If only one view, we do not have edge case
	{
		thrust::copy(GF.begin() + YL * ZL, GF.begin() + 2 * YL * ZL, GF.begin());
		thrust::copy(GF.begin() + (ProjNumber - 2) * YL * ZL,
			GF.begin() + (ProjNumber - 1) * YL * ZL,
			GF.begin() + (ProjNumber - 1) * YL * ZL);
	}
}



// Compute the derivation of u
template<typename T>
__global__ void DevAlongU_ker(T* GF, const T* Proj, const T* coef, const int YL, const int ZL, const int ProjNumber)
{
	int yIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int zIdx = threadIdx.y + blockIdx.y * blockDim.y;
	int pIdx = threadIdx.z + blockIdx.z * blockDim.z;
	if (yIdx < YL - 2 && yIdx > 1 && zIdx < ZL && pIdx < ProjNumber)
	{
		int lidx = (pIdx * ZL + zIdx) * YL + yIdx - 1;
		int ridx = (pIdx * ZL + zIdx) * YL + yIdx + 1;
		int idx = (pIdx * ZL + zIdx) * YL + yIdx;
		GF[idx] += (Proj[ridx] - Proj[lidx]) * coef[yIdx];
	}
}

template<typename T>
__global__ void getTempAlongUfirst(T* res, const T* Proj, const int YL, const int ZL, const int ProjNumber)
{
	int zIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int pIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if (zIdx < ZL && pIdx < ProjNumber)
	{
		res[pIdx * ZL + zIdx] = Proj[(pIdx * ZL + zIdx) * YL + 2] - Proj[(pIdx * ZL + zIdx) * YL];
	}
}
template<typename T>
__global__ void getTempAlongUlast(T* res, const T* Proj, const int YL, const int ZL, const int ProjNumber)
{
	int zIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int pIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if (zIdx < ZL && pIdx < ProjNumber)
	{
		res[pIdx * ZL + zIdx] = Proj[(pIdx * ZL + zIdx) * YL + YL - 1] - Proj[(pIdx * ZL + zIdx) * YL + YL - 3];
	}
}
template<typename T>
__global__ void setTempAlongU(T* GF, const T* temp, int yIndex, const T coef, const int YL, const int ZL, const int ProjNumber)
{
	int zIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int pIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if (zIdx < ZL && pIdx < ProjNumber)
	{
		GF[(pIdx * ZL + zIdx) * YL + yIndex] += temp[pIdx * ZL + zIdx] * coef;
	}
}

template<typename T>
void DevAlongU_gpu(thrust::device_vector<T>& GF, const thrust::device_vector<T>& Proj,
	const int YL, const int ZL, const int ProjNumber, const T StdDisSquare,
	const thrust::host_vector<T>& hyCor,
	const T invTWODeltaUxStdDis)
{
	thrust::host_vector<T> hcoef(YL);
	//thrust::transform(hyCor.begin(), hyCor.end(), hcoef.begin(), [&](T ycor) { return (StdDisSquare + ycor * ycor) * invTWODeltaUxStdDis});
	for (int i = 0; i != YL; ++i)
	{
		hcoef[i] = (StdDisSquare + hyCor[i] * hyCor[i]) * invTWODeltaUxStdDis;
	}
	thrust::device_vector<T> coef = hcoef;
	//thrust::transform(yCor.begin(), yCor.end(), coef.begin(), [&](T y){ return (StdDisSquare + y * y) * invTWODeltaUxStdDis;});
	dim3 blk(64, 8, 2);
	dim3 gid((YL + blk.x - 1) / blk.x,
		(ZL + blk.y - 1) / blk.y,
		(ProjNumber + blk.z - 1) / blk.z);
	DevAlongU_ker<T> << <gid, blk >> >(
		thrust::raw_pointer_cast(&GF[0]),
		thrust::raw_pointer_cast(&Proj[0]),
		thrust::raw_pointer_cast(&coef[0]),
		YL, ZL, ProjNumber);
	dim3 blk1(16, 64);
	dim3 gid1(
		(ZL + blk1.x - 1) / blk1.x,
		(ProjNumber + blk1.y - 1) / blk1.y);
	thrust::device_vector<T> temp1(ZL * ProjNumber, 0);
	thrust::device_vector<T> temp2(ZL * ProjNumber, 0);
	getTempAlongUfirst<T> << <gid1, blk1 >> >(thrust::raw_pointer_cast(&temp1[0]),
		thrust::raw_pointer_cast(&Proj[0]), YL, ZL, ProjNumber);
	T coef0 = (StdDisSquare + hyCor[0] * hyCor[0]) * invTWODeltaUxStdDis;
	T coef1 = (StdDisSquare + hyCor[1] * hyCor[1]) * invTWODeltaUxStdDis;
	setTempAlongU << <gid1, blk1 >> >(thrust::raw_pointer_cast(&GF[0]), thrust::raw_pointer_cast(&temp1[0]), 0, coef0, YL, ZL, ProjNumber);
	setTempAlongU << <gid1, blk1 >> >(thrust::raw_pointer_cast(&GF[0]), thrust::raw_pointer_cast(&temp1[0]), 1, coef1, YL, ZL, ProjNumber);

	getTempAlongUlast<T> << <gid1, blk1 >> >(thrust::raw_pointer_cast(&temp2[0]),
		thrust::raw_pointer_cast(&Proj[0]), YL, ZL, ProjNumber);
	T coef_1 = (StdDisSquare + hyCor[YL - 1] * hyCor[YL - 1]) * invTWODeltaUxStdDis;
	T coef_2 = (StdDisSquare + hyCor[YL - 2] * hyCor[YL - 2]) * invTWODeltaUxStdDis;
	setTempAlongU << <gid1, blk1 >> >(thrust::raw_pointer_cast(&GF[0]), thrust::raw_pointer_cast(&temp2[0]), YL - 2, coef_2, YL, ZL, ProjNumber);
	setTempAlongU << <gid1, blk1 >> >(thrust::raw_pointer_cast(&GF[0]), thrust::raw_pointer_cast(&temp2[0]), YL - 1, coef_1, YL, ZL, ProjNumber);
}



// Compute the derivative of v
template<typename T>
__global__ void DevAlongV_ker(T* GF, const T* Proj, const T* y, const T* z, const T coef, const int YL, const int ZL, const int ProjNumber)
{
	int yIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int zIdx = threadIdx.y + blockIdx.y * blockDim.y;
	int pIdx = threadIdx.z + blockIdx.z * blockDim.z;
	if (yIdx < YL && zIdx > 1 && zIdx < ZL - 2 && pIdx < ProjNumber)
	{
		int idx = (pIdx * ZL + zIdx) * YL + yIdx;
		int rIdx = (pIdx * ZL + zIdx + 1) * YL + yIdx;
		int lIdx = (pIdx * ZL + zIdx - 1) * YL + yIdx;

		const T Y = y[yIdx];
		const T Z = z[zIdx];
		const T temp = Y * (Proj[rIdx] - Proj[lIdx]);
		GF[idx] += Z * temp * coef;
	}
}


template<typename T>
__global__ void getTempAlongVfirst(T* temp, const T* Proj, const T* y, const int YL, const int ZL, const int ProjNumber)
{
	int yIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int pIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if (yIdx < YL && pIdx < ProjNumber)
	{
		temp[pIdx * YL + yIdx] = (Proj[(pIdx * ZL + 2) * YL + yIdx] - Proj[(pIdx * ZL + 0) * YL + yIdx]) * y[yIdx];
	}
}
template<typename T>
__global__ void getTempAlongVlast(T* temp, const T* Proj, const T* y, const int YL, const int ZL, const int ProjNumber)
{
	int yIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int pIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if (yIdx < YL && pIdx < ProjNumber)
	{
		temp[pIdx * YL + yIdx] = (Proj[(pIdx * ZL + ZL - 1) * YL + yIdx] - Proj[(pIdx * ZL + ZL - 3) * YL + yIdx]) * y[yIdx];
	}
}

template<typename T>
__global__ void setTempAlongV(T* GF, const T* temp, int zIndex, const T zxcoef, const int YL, const int ZL, const int ProjNumber)
{
	int yIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int pIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if (yIdx < YL && pIdx < ProjNumber)
	{
		GF[(pIdx * ZL + zIndex) * YL + yIdx] += temp[pIdx * YL + yIdx] * zxcoef;
	}
}


template<typename T>
void DevAlongV_gpu(thrust::device_vector<T>& GF, const thrust::device_vector<T>& Proj,
	const thrust::host_vector<T>& hy, const thrust::host_vector<T>& hz, const T DeltaV, const T StdDis,
	const int YL, const int ZL, const int ProjNumber)
{

	const thrust::device_vector<T> y = hy;
	const thrust::device_vector<T> z = hz;

	const T coef = 1.0 / (2.0 * DeltaV * StdDis);
	dim3 blk(64, 8, 2);
	dim3 gid((YL + blk.x - 1) / blk.x,
		(ZL + blk.y - 1) / blk.y,
		(ProjNumber + blk.z - 1) / blk.z);
	DevAlongV_ker<T> << <gid, blk >> >(thrust::raw_pointer_cast(&GF[0]),
		thrust::raw_pointer_cast(&Proj[0]),
		thrust::raw_pointer_cast(&y[0]),
		thrust::raw_pointer_cast(&z[0]),
		coef, YL, ZL, ProjNumber);
	thrust::device_vector<T> temp1(YL * ProjNumber, 0);
	thrust::device_vector<T> temp2(YL * ProjNumber, 0);
	dim3 blk1(64, 16);
	dim3 gid1((YL + blk1.x - 1) / blk1.x,
		(ProjNumber + blk1.y - 1) / blk1.y);

	getTempAlongVfirst<T> << <gid1, blk1 >> >(thrust::raw_pointer_cast(&temp1[0]), thrust::raw_pointer_cast(&Proj[0]), thrust::raw_pointer_cast(&y[0]), YL, ZL, ProjNumber);
	setTempAlongV<T> << <gid1, blk1 >> >(thrust::raw_pointer_cast(&GF[0]), thrust::raw_pointer_cast(&temp1[0]), 0, hz[0] * coef, YL, ZL, ProjNumber);
	setTempAlongV<T> << <gid1, blk1 >> >(thrust::raw_pointer_cast(&GF[0]), thrust::raw_pointer_cast(&temp1[0]), 1, hz[1] * coef, YL, ZL, ProjNumber);

	getTempAlongVlast<T> << <gid1, blk1 >> >(thrust::raw_pointer_cast(&temp2[0]), thrust::raw_pointer_cast(&Proj[0]), thrust::raw_pointer_cast(&y[0]), YL, ZL, ProjNumber);
	setTempAlongV<T> << <gid1, blk1 >> >(thrust::raw_pointer_cast(&GF[0]), thrust::raw_pointer_cast(&temp2[0]), ZL - 2, hz[ZL - 2] * coef, YL, ZL, ProjNumber);
	setTempAlongV<T> << <gid1, blk1 >> >(thrust::raw_pointer_cast(&GF[0]), thrust::raw_pointer_cast(&temp2[0]), ZL - 1, hz[ZL - 1] * coef, YL, ZL, ProjNumber);
}

/////////////////////////////////////////////////////////////////////////////////////////////

// AZCor[AL][YL]
template<typename T>
__global__ void resamplingProjection(T* ConvRes, const T* GF, const T* AZCor, const T* coef,
	const T DeltaV, const T HalfZ, const int YL, const int ZL, const int AL, const int ProjNumber)
{
	int tempindex = threadIdx.x + blockIdx.x * blockDim.x; //YL
	int AIndex = threadIdx.y + blockIdx.y * blockDim.y; // AL
	int pIdx = threadIdx.z + blockIdx.z * blockDim.z; // ProjNumber
	if (tempindex < YL && AIndex < AL && pIdx < ProjNumber)
	{
		T tempV = AZCor[AIndex * YL + tempindex] / DeltaV + HalfZ;
		int idx = (pIdx * AL + AIndex) * YL + tempindex;
		T coefv = coef[AIndex * YL + tempindex];

		if (tempV < 0.01)
		{
			ConvRes[idx] = GF[(pIdx * ZL + 0) * YL + tempindex] * coefv;
		}
		else if (tempV >(ZL - 1))
		{
			ConvRes[idx] = GF[(pIdx * ZL + ZL - 1) * YL + tempindex] * coefv;
		}
		else
		{
			int tpUpv = ceil(tempV);
			int tpLwv = tpUpv - 1;
			ConvRes[idx] = (GF[(pIdx * ZL + tpUpv) * YL + tempindex] * (tempV - tpLwv)
				+ GF[(pIdx * ZL + tpLwv) * YL + tempindex] * (tpUpv - tempV)) * coefv;
		}
	}
}


template<typename T>
thrust::host_vector<thrust::host_vector<T>> Resampling(
	thrust::device_vector<T>& d_GF,
	const thrust::device_vector<T>& d_Proj,
	const T HalfZ, const T DeltaV, const T RebinFactor, const T delta,
	const T FilterCoeFF, const T HelicP, const T ScanR, const T StdDis,
	const int FilteringMode, const thrust::host_vector<T>& y,
	const int YL, const int ZL, const int ProjNumber, const int DecHeigh, const int AL)
{
	const double PI = 3.14159265358979323846264;
	const double DeltaAngle = delta;
	const double AngleRange = 2.0 * PI - DeltaAngle;
	const int QQ = ceil(ZL * RebinFactor * 0.5);
	thrust::host_vector<T> Nata(AL, 0);
	thrust::host_vector<T> FilterCoefVec(AL, 0);
	thrust::host_vector<T> AA(AL, 0);
	thrust::host_vector<T> BB(AL, 0);
	thrust::host_vector<T> CC(AL, 0);

	for (int i = 0; i != AL; ++i)
	{
		Nata[i] = (i - QQ) * AngleRange / (static_cast<double>(QQ) - 1.0);
	}

	if (FilteringMode == 2)
	{
		for (int i = 0; i != AL; ++i)
		{
			FilterCoefVec[i] = FilterCoeFF;
		}
		FilterCoefVec[QQ] = 0;
	}
	else
	{
		for (int i = 0; i != QQ; ++i)
		{
			FilterCoefVec[i] = 1.0 - FilterCoeFF;
		}
		for (int i = QQ + 1; i != AL; ++i)
		{
			FilterCoefVec[i] = FilterCoeFF;
		}
		FilterCoefVec[QQ] = 0;
	}


	for (int i = 0; i != AL; ++i)
	{
		AA[i] = sin(FilterCoefVec[i] * Nata[i]) - FilterCoefVec[i] * sin(Nata[i]);
		BB[i] = FilterCoefVec[i] - 1.0 + cos(FilterCoefVec[i] * Nata[i]) - FilterCoefVec[i] * cos(Nata[i]);
		CC[i] = sin((1.0 - FilterCoefVec[i]) * Nata[i]) - sin(Nata[i]) + sin(FilterCoefVec[i] * Nata[i]);

	}
	CC[QQ] = 1.0;

	// inter distance
	thrust::host_vector<T> InDs(AL, 0);
	thrust::host_vector<T> Rato(AL, 0);
	thrust::host_vector<T> AZCor(AL * YL, 0);
	for (int i = 0; i != AL; ++i)
	{
		InDs[i] = Nata[i] * HelicP * StdDis * AA[i] / (2.0 * PI * ScanR * CC[i]);
		Rato[i] = Nata[i] * HelicP * BB[i] / (2.0 * PI * ScanR * CC[i]);
	}
	Rato[QQ] = HelicP / (2.0 * PI * ScanR);

	thrust::host_vector<T> coef(AL * YL, 0);

	thrust::host_vector<T> zmax(AL * YL, -DecHeigh);
	thrust::host_vector<T> zmin(AL * YL, DecHeigh);
	thrust::host_vector<T> MAXI(AL * YL, 1);
	thrust::host_vector<T> MINI(AL * YL, AL);


	for (int i = 0; i != AL; ++i) // AZCor[AL][YL]
	{
		for (int j = 0; j != YL; ++j)
		{
			int idx = i * YL + j;
			AZCor[idx] = InDs[i] + Rato[i] * y[j];
			coef[idx] = StdDis / sqrt(StdDis * StdDis + y[j] * y[j] + AZCor[idx] * AZCor[idx]);

			if (AZCor[idx] > zmax[idx])
			{
				zmax[idx] = AZCor[idx];
				MAXI[idx] = i;
			}
			if (AZCor[idx] < zmin[idx])
			{
				zmin[idx] = AZCor[idx];
				MINI[idx] = i;
			}
		}
	}

	thrust::device_vector<T> ConvRes(ProjNumber * AL * YL, 0);
	thrust::device_vector<T> d_AZCor = AZCor;
	thrust::device_vector<T> d_coef = coef;
	dim3 blk(32, 8, 2);
	dim3 gid((YL + blk.x - 1) / blk.x,
		(AL + blk.y - 1) / blk.y,
		(ProjNumber + blk.z - 1) / blk.z);
	resamplingProjection<T> << <gid, blk >> >(thrust::raw_pointer_cast(&ConvRes[0]),
		thrust::raw_pointer_cast(&d_GF[0]),
		thrust::raw_pointer_cast(&d_AZCor[0]),
		thrust::raw_pointer_cast(&d_coef[0]),
		DeltaV, HalfZ, YL, ZL, AL, ProjNumber);


	thrust::host_vector<thrust::host_vector<T>> res;
	res.push_back(coef);
	res.push_back(zmax);
	res.push_back(zmin);
	res.push_back(MAXI);
	res.push_back(MINI);
	res.push_back(AZCor);
	res.push_back(Rato);
	d_GF = ConvRes;
	return res;
}



/// Round up to next higher power of 2 (return x if it's already a power
/// of 2).
inline int nextpow2(int x)
{
	if (x < 0)
		return 0;
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x + 1;
}

// TODO: In this case, we only use the rectangle window
enum FilterWindowType{RECTANGLE, KAISER, HAMMING, HANN, BLACKMAN};


template<typename T>
thrust::host_vector<T> CreateHSP(int XS, FilterWindowType filteringType)
{
	int Length = XS;
	thrust::host_vector<T> HS(Length, 1.0);
	const int Center = Length * 0.5 + 1.0;
	const T PI = 3.14159265358979323846264;
	HS[0] = 0;
	for (int i = 2; i <= Center - 1; ++i)
	{
		T temp = (static_cast<double>(i) - Center) * PI;
		HS[i - 1] = 2.0 * pow(sin(temp / 2.0), 2.0) / (temp);
	}

	for (int i = Center + 1; i <= Length; ++i)
	{
		T temp = (static_cast<double>(i) - Center) * PI;
		HS[i - 1] = 2.0 * pow(sin(temp / 2.0), 2.0) / (temp);
	}
	HS[Center - 1] = 0;

    thrust::host_vector<T> W(Length, 1.0);
    //switch (filteringType)
    //{
    //case RECTANGLE:
    //    break;
    //case KAISER:
    //    // We do not support KAISER yet;
    //case HAMMING:
    //    int N = XS - 1;
    //    for (int i = 0; i <= N; ++i) {
    //        W[i] = 0.54 - 0.46 * cos(2.0 * PI * static_cast<T>(i) / static_cast<int>(N));
    //    }
    //    break;
    //case HANN:
    //    int N = XS - 1;
    //    for (int i = 0; i <= N; ++i) {
    //        W[i] = 0.5 - cos(2.0 * PI * static_cast<T>(i) / static_cast<int>(N));
    //    }
    //    break;
    //case BLACKMAN:
    //    int M = XS / 2;
    //    if (XS % 2) {
    //        M = (XS+1)/2
    //    }
    //    else {
    //        M = XS / 2;
    //    }
    //    for (int i = 0; i <= (M - 1); ++i) {
    //        double ww = 0.42 - 0.5 * cos(2.0 * PI * static_cast<T>(i) / static_cast(XS - 1)) + 0.08 * cos(4.0 * PI * static_cast<T>(i) / static_cast<T>(N - 1));
    //        W[i] = ww;
    //        W[W.size() - 1 - i] = ww;
    //    }
    //    break;
    //default:
    //    break;

    //}

    for (int i = 0; i < HS.size(); ++i) {
        HS[i] *= W[i];
    }

	return HS;
}


template<typename T>
__global__ void copyHSKer(cufftComplex* d_HSForFFT, const T* dev_HS, const int L)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < L)
	{
		d_HSForFFT[idx].x = dev_HS[idx];
		d_HSForFFT[idx].y = 0.0;
	}
}

template<typename T>
__global__ void expandProj(cufftComplex* output, const T* input, const int YL, const int ZL, const int ProjNumber, const int nn2)
{
	int curPos = threadIdx.x + blockIdx.x * blockDim.x;
	int curBatch = threadIdx.y + blockIdx.y * blockDim.y;
	if (curPos < YL && curBatch < ZL * ProjNumber)
	{
		output[curBatch * nn2 + curPos].x = input[curBatch * YL + curPos];
		output[curBatch * nn2 + curPos].y = 0.0;
	}
}


__global__ void multiProjWithKer(cufftComplex* exProj, const cufftComplex* kernel, const int nn2, const int TotalNum)
{
	int kerIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int batIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if (kerIdx < nn2 && batIdx < TotalNum)
	{
		cufftComplex ker = kernel[kerIdx];
		cufftComplex pj = exProj[batIdx * nn2 + kerIdx];
		cufftComplex res;
		res.x = pj.x * ker.x - pj.y * ker.y;
		res.y = pj.x * ker.y + pj.y * ker.x;
		exProj[batIdx * nn2 + kerIdx] = res;
	}

}

template<typename T>
__global__ void cutProjData(T* toReturn, const cufftComplex* proj, const T* coef, const int YL, const int nn2, const int ProjNumber, const int AL)
{
	int yIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int zIdx = threadIdx.y + blockIdx.y * blockDim.y;
	int pIdx = threadIdx.z + blockIdx.z * blockDim.z;
	if (yIdx < YL && zIdx < AL && pIdx < ProjNumber)
	{
		int inIdx = (pIdx * AL + zIdx) * nn2 + yIdx;
		int outIdx = (pIdx * AL + zIdx) * YL + yIdx;
		int cIdx = zIdx * YL + yIdx;

		toReturn[outIdx] = -proj[inIdx].x / static_cast<T>(nn2) / coef[cIdx];
	}

}


template<typename T>
void HiltertWeightedFilter(thrust::device_vector<T>& ConvRes, const thrust::device_vector<T>& d_coef, const int YL, const int AL, const int ProjNumber)
{
	const int nn = pow(2.0, log2(static_cast<T>(nextpow2(YL))) + 1);
	const int nn2 = 2 * nn;
    FilterWindowType filterWindwoType = RECTANGLE;
	thrust::host_vector<T> tempHS = CreateHSP<T>(nn, filterWindwoType); // This can be exposed as a parameter
	thrust::host_vector<T> HS(nn2, 0);
	thrust::copy(tempHS.begin() + nn / 2, tempHS.begin() + nn, HS.begin());
	thrust::copy(tempHS.begin(), tempHS.begin() + nn / 2, HS.begin() + nn + nn / 2);
	tempHS.clear();

	for (int i = 0; i != nn2; ++i)
	{
		if (abs(HS[i]) < 1.0E-20)
		{
			HS[i] = 0;
		}
	}
	thrust::device_vector<T> dev_HS = HS;
	HS.clear();
	//Calculate FFT of HS

	dim3 blkCopyHSFFT(1024);
	dim3 gidCopyHSFFT((nn2 + blkCopyHSFFT.x - 1) / blkCopyHSFFT.x);
	thrust::device_vector<cufftComplex> d_HSForFFT(nn2);
	copyHSKer<T> << <gidCopyHSFFT, blkCopyHSFFT >> >(
		thrust::raw_pointer_cast(&d_HSForFFT[0]),
		thrust::raw_pointer_cast(&dev_HS[0]), nn2);
	cufftHandle planHS;
	cufftPlan1d(&planHS, nn2, CUFFT_C2C, 1);
	cufftExecC2C(planHS, thrust::raw_pointer_cast(&d_HSForFFT[0]), thrust::raw_pointer_cast(&d_HSForFFT[0]), CUFFT_FORWARD);
	cufftDestroy(planHS);

	// Expand the projection data
	thrust::device_vector<cufftComplex> expProj(YL * nn2 * ProjNumber);
	dim3 blkExp(128, 8);
	dim3 gidExp(
		(YL + blkExp.x - 1) / blkExp.x,
		(AL * ProjNumber + blkExp.y - 1) / blkExp.y);
	expandProj<T> << <gidExp, blkExp >> >(
		thrust::raw_pointer_cast(&expProj[0]),
		thrust::raw_pointer_cast(&ConvRes[0]), YL, AL, ProjNumber, nn2);

	ConvRes.clear();

	cufftHandle plan;
	cufftPlan1d(&plan, nn2, CUFFT_C2C, AL * ProjNumber);
	cufftExecC2C(plan, thrust::raw_pointer_cast(&expProj[0]),
		thrust::raw_pointer_cast(&expProj[0]), CUFFT_FORWARD);

	// Multiply with Kernel
	dim3 multiBlk(64, 8);
	dim3 multiGid(
		(nn2 + multiBlk.x - 1) / multiBlk.x,
		(AL * ProjNumber + multiBlk.y - 1) / multiBlk.y);

	multiProjWithKer << <multiGid, multiBlk >> >(
		thrust::raw_pointer_cast(&expProj[0]),
		thrust::raw_pointer_cast(&d_HSForFFT[0]), nn2, AL * ProjNumber);

	d_HSForFFT.clear();

	// back batch FFT
	cufftExecC2C(plan, thrust::raw_pointer_cast(&expProj[0]),
		thrust::raw_pointer_cast(&expProj[0]), CUFFT_INVERSE);

	// Cut the data
	ConvRes.clear();
	ConvRes.resize(YL * AL * ProjNumber);

	dim3 blkCut(64, 8, 2);
	dim3 gidCut(
		(YL + blkCut.x - 1) / blkCut.x,
		(AL + blkCut.y - 1) / blkCut.y,
		(ProjNumber + blkCut.z - 1) / blkCut.z);
	cutProjData<T> << <gidCut, blkCut >> >(thrust::raw_pointer_cast(&ConvRes[0]),
		thrust::raw_pointer_cast(&expProj[0]),
		thrust::raw_pointer_cast(&d_coef[0]),
		YL, nn2, ProjNumber, AL);

	cufftDestroy(planHS);
	cufftDestroy(plan);
	expProj.clear();
}


template<typename T, int batSize>
void HiltertFiltering(thrust::device_vector<T>& d_GF, const thrust::device_vector<T>& d_coef, const int YL, const int AL, const int ProjNumber)
{
	int GroupNum = ceil(ProjNumber / batSize);
	for (int ii = 0; ii != GroupNum - 1; ++ii)
	{
		thrust::device_vector<T> bat(d_GF.begin() + ii * YL * AL * batSize, d_GF.begin() + (ii + 1) * YL * AL * batSize);
		HiltertWeightedFilter<T>(bat, d_coef, YL, AL, batSize);
		thrust::copy(bat.begin(), bat.end(), d_GF.begin() + ii * YL * AL * batSize);
	}
	thrust::device_vector<T> bat(d_GF.begin() + (GroupNum - 1) * YL * AL * batSize, d_GF.end());
	HiltertWeightedFilter<T>(bat, d_coef, YL, AL, (ProjNumber - (GroupNum - 1) * batSize));
	thrust::copy(bat.begin(), bat.end(), d_GF.begin() + (GroupNum - 1) * YL * AL * batSize);
}


template<typename T>
__global__ void RebinningFilteredResults_ker(T* FProj, const T* ConvRes,const T* zmaxV, const T* zminV, const T* MAXIV, const T* MINIV,
	const T* AZCor, // AZCor[AL][YL]
	const T* Rato, const T* y,
	const int YL, const int ZL, const int AL, const int ProjNum, const int QQ,
	const T HalfZ, const T DeltaV, const T DecHeigh)
{
	int yIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int zIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if (yIdx < YL && zIdx < ZL)
	{
		//int tIdx = zIdx * YL + yIdx;
		//T zmin = zminV[tIdx];
		//T zmax = zmaxV[tIdx];
		//T MINI = MINIV[tIdx];
		//T MAXI = MAXIV[tIdx];

        T zmax = -DecHeigh;
        T zmin = DecHeigh;
        T MAXI = 1;
        T MINI = AL;

        for (int tpindex = 0; tpindex < AL; ++tpindex) {
            if (AZCor[tpindex * YL + yIdx] > zmax) {
                zmax = AZCor[tpindex * YL + yIdx];
                MAXI = tpindex;
            }
            if (AZCor[tpindex * YL + yIdx] < zmin) {
                zmin = AZCor[tpindex * YL + yIdx];
                MINI = tpindex;
            }
        }

		int BeginAL = 0;
		int EndAL = 0;
		T BeginCoef = 0;
		T EndCoef = 0;
		T CurrZ = (static_cast<T>(zIdx) - HalfZ) * DeltaV;
		if (CurrZ < zmin)
		{
			BeginAL = MINI;
			EndAL = MINI;
			BeginCoef = 0.5;
			EndCoef = 0.5;
		}
		else if (CurrZ > zmax)
		{
			BeginAL = MAXI;
			EndAL = MAXI;
			BeginCoef = 0.5;
			EndCoef = 0.5;
		}
		else
		{
			T deltamaxB = DecHeigh;
			T deltamaxE = DecHeigh;
			for (int tpindex = 0; tpindex != AL; ++tpindex) 
			{
				if (AZCor[tpindex * YL + yIdx] > CurrZ && AZCor[tpindex * YL + yIdx] - CurrZ < deltamaxE)
				{
					deltamaxE = AZCor[tpindex * YL + yIdx] - CurrZ;
					EndAL = tpindex;
				}
				if (AZCor[tpindex * YL + yIdx] < CurrZ && CurrZ - AZCor[tpindex * YL + yIdx] < deltamaxB)
				{
					deltamaxB = CurrZ - AZCor[tpindex * YL + yIdx];
					BeginAL = tpindex;
				}
			}


			if (!((EndAL == BeginAL + 1) || (EndAL == BeginAL)))
			{
				double Sign_Dis = CurrZ - Rato[QQ + 1] * y[yIdx];
				if (Sign_Dis > 0)
				{
					if (BeginAL < QQ + 1)
					{
						BeginAL = EndAL - 1;
						deltamaxB = CurrZ - AZCor[BeginAL * YL + yIdx];
					}
					else if (EndAL < QQ + 1)
					{
						EndAL = BeginAL + 1;
						deltamaxE = AZCor[EndAL * YL + yIdx] - CurrZ;
					}
					else if (EndAL < BeginAL)
					{
						BeginAL = EndAL - 1;
						deltamaxB = CurrZ - AZCor[BeginAL * YL + yIdx];
					}
					else {
						EndAL = BeginAL + 1;
						deltamaxE = AZCor[EndAL * YL + yIdx] - CurrZ;
					}
				}
				else
				{
					if (BeginAL > QQ + 1)
					{
						BeginAL = EndAL - 1;
						deltamaxB = CurrZ - AZCor[BeginAL * YL + yIdx];
					}
					else if (EndAL > QQ + 1)
					{
						EndAL = BeginAL + 1;
						deltamaxE = AZCor[EndAL * YL + yIdx] - CurrZ;
					}
					else if (EndAL > BeginAL)
					{
						BeginAL = EndAL - 1;
						deltamaxB = CurrZ - AZCor[BeginAL * YL + yIdx];
					}
					else
					{
						EndAL = BeginAL + 1;
						deltamaxE = AZCor[EndAL * YL + yIdx] - CurrZ;
					}
				}
			}

			BeginCoef = deltamaxE / (deltamaxE + deltamaxB);
			EndCoef = deltamaxB / (deltamaxE + deltamaxB);
		}

		for (int pIdx = 0; pIdx != ProjNum; ++pIdx)
		{
            FProj[(pIdx * ZL + zIdx) * YL + yIdx] = ConvRes[(pIdx * AL + BeginAL) * YL + yIdx] * BeginCoef + ConvRes[(pIdx * AL + EndAL) * YL + yIdx] * EndCoef;
		}
	}
}

template<typename T>
thrust::device_vector<T> RebinningFilteredResults(
	const thrust::device_vector<T>& ConvRes, 
	const thrust::device_vector<T>& zmaxV, 
	const thrust::device_vector<T>& zminV, 
	const thrust::device_vector<T>& MAXIV, 
	const thrust::device_vector<T>& MINIV,
	const thrust::device_vector<T>& AZCor, // AZCor[AL][YL]
	const thrust::device_vector<T>& Rato, 
	const thrust::device_vector<T>& y,
	const int YL, const int ZL, const int AL, const int ProjNum, const int QQ,
	const T HalfZ, const T DeltaV, const T DecHeigh, const int yDim, const int zDim)
{
	thrust::device_vector<T> FProj(YL * ZL * ProjNum, 0);
	dim3 blk(yDim, zDim);
	dim3 gid(
		(YL + yDim - 1) / yDim,
		(ZL + zDim - 1) / zDim);

	RebinningFilteredResults_ker<<<gid,blk>>>(
		thrust::raw_pointer_cast(&FProj[0]),
		thrust::raw_pointer_cast(&ConvRes[0]),
		thrust::raw_pointer_cast(&zmaxV[0]),
		thrust::raw_pointer_cast(&zminV[0]),
		thrust::raw_pointer_cast(&MAXIV[0]),
		thrust::raw_pointer_cast(&MINIV[0]),
		thrust::raw_pointer_cast(&AZCor[0]),
		thrust::raw_pointer_cast(&Rato[0]),
		thrust::raw_pointer_cast(&y[0]),
		YL, ZL, AL, ProjNum, QQ, HalfZ, DeltaV, DecHeigh);
	return FProj;
}


template<typename T>
thrust::host_vector<T> KatsevichFiltering_GPU(const thrust::host_vector<T>& Proj,
	const int ProjScale, 			// Number of views per rotation
	const T DecWidth, 			// width of the detector array
	const T DecHeigh, 			// height of the detector array
	const T ScanR,    			// scan radius
	const T StdDis,            // source to detector distance
	const T HelicP,   			// Helical pitch
	const T FilterCoeFF, 		// filtering coefficient
	const T delta,    			// 2 * acos(ObjR / ScanR);
	const int FilteringMode,		// filtering mode == 2  or others
	const int YL,          			// detector cell along transverse direction
	const int ZL,        			// detector cell along bench moving direction
	const int ProjNumber)  			// number of views totally
{
	const T RebinFactor = 1.3;
	const int QQ = ceil(ZL*RebinFactor*0.5);
	const int AL = 2 * QQ + 1;
	//const double ParaCoef = FilterCoeFF;
	//const int WindowType = 1; // 1 rectangle window; 2 kaiser window; 3 hamming window; 4 hanning window; 5 blackman window


	const T DeltaL = 2.0 * PI / static_cast<T>(ProjScale); // projection view step
	const T DeltaU = DecWidth / static_cast<T>(YL);
	const T DeltaV = DecHeigh / static_cast<T>(ZL);
	const T HalfZ = (ZL - 1.0) * 0.5;
	const T HalfY = (YL - 1.0) * 0.5;

	// Generate y coordinates
	thrust::host_vector<T> y(YL, 0);
	thrust::host_vector<T> z(ZL, 0);

	int i = 0;
	thrust::transform(y.begin(), y.end(), y.begin(), [&](T yy) { return (i++ - HalfY) * DeltaU; });
	i = 0;
	thrust::transform(z.begin(), z.end(), z.begin(), [&](T zz) { return (i++ - HalfZ) * DeltaV; });

	// Step 1 Compute the derivative of projections
	thrust::device_vector<T> d_Proj = Proj;
	thrust::device_vector<T> d_GF = d_Proj;

	DevAlongProjview_gpu<T>(d_GF, d_Proj, YL, ZL, ProjNumber, DeltaL);
    
	// Compute the derivation of U
	const T StdDisSquare = StdDis * StdDis;
	const T invTWODeltaUxStdDis = 1.0 / (2.0 * DeltaU * StdDis);

	DevAlongU_gpu<T>(d_GF, d_Proj, YL, ZL, ProjNumber, StdDisSquare, y, invTWODeltaUxStdDis);

	// Compute derivative of V
	DevAlongV_gpu<T>(d_GF, d_Proj, y, z, DeltaV, StdDis, YL, ZL, ProjNumber);
    
	// Resampling from the projections and weighted before the filtered procedure
	thrust::host_vector<thrust::host_vector<T>> res = Resampling<T>(d_GF, d_Proj, HalfZ, DeltaV, RebinFactor, delta, FilterCoeFF, HelicP, ScanR, StdDis, FilteringMode, y, YL, ZL, ProjNumber, DecHeigh, AL);
	thrust::device_vector<T> d_coef = res[0];//[AL][YL] 
	thrust::device_vector<T> zmax = res[1];
	thrust::device_vector<T> zmin = res[2];
	thrust::device_vector<T> MAXI = res[3];
	thrust::device_vector<T> MINI = res[4];
	thrust::device_vector<T> AZCor = res[5]; //[AL][YL]
	thrust::device_vector<T> Rato = res[6];
    
	thrust::device_vector<T> d_y = y;

	// Hiltert transform and weighted after the filtered procedure
	HiltertFiltering<T, 1024>(d_GF, d_coef, YL, AL, ProjNumber);

	int yDim = 32;
	int zDim = 8;

	thrust::device_vector<T> FProj = RebinningFilteredResults(d_GF, zmax, zmin, MAXI, MINI,
		AZCor, 
		Rato,
		d_y,
		YL, ZL, AL, ProjNumber, QQ, HalfZ, DeltaV, DecHeigh, yDim, zDim);
	thrust::host_vector<T> toReturn = FProj;
	return toReturn;
}



void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {

    double* projPtr = mxGetPr(prhs[0]);

    double*　ProjScalePtr = mxGetPr(prhs[1]);
    double* DecWidth = mxGetPr(prhs[2]);
    double* DecHeigh = mxGetPr(prhs[3]);
    double* ScanR = mxGetPr(prhs[4]);
    double* StdDis = mxGetPr(prhs[5]);
    double* HelicP = mxGetPr(prhs[6]);
    double* FilterCoeFF = mxGetPr(prhs[7]);
    double* DeltaAngle = mxGetPr(prhs[8]);
    double* FilteringMode = mxGetPr(prhs[9]);
    double* YL = mxGetPr(prhs[10]);
    double* ZL = mxGetPr(prhs[11]);
    double* ProjNumber = mxGetPr(prhs[12]);

    double* Res = mxGetPr(prhs[13]);
    int YLL = *YL;
    int ZLL = *ZL;
    int ProjNumm = *ProjNumber;
    thrust::host_vector<double> Proj(projPtr, projPtr + YLL * ZLL * ProjNumm);

    thrust::host_vector<double> res = KatsevichFiltering_GPU<double>(Proj,
        *ProjScalePtr,
        *DecWidth,
        *DecHeigh,
        *ScanR,
        *StdDis,
        *HelicP,
        *FilterCoeFF,
        *DeltaAngle,
        *FilteringMode, 
        *YL, 
        *ZL, 
        *ProjNumber);

    thrust::copy(&(res[0]), &(res[0]) + YLL * ZLL * ProjNumm, Res);

}

//
//
//
//int main()
//{
//	cudaSetDevice(1);
//	
//
//
//	thrust::host_vector<double> Proj(300 * 40 * 3502, 0);
//	std::ifstream fin("C:\\Users\\liuru\\OneDrive\\Documents\\Visual Studio 2015\\Projects\\KatsevichFiltering_GPU\\KatsevichFiltering_GPU\\Katsevich_ALgorihtm_Code\\intermediateData\\OriginalProjectionData_300x40x3502.data", std::ios::binary);
//	if (!fin.is_open())
//	{
//		std::cout << "Cannot open file\n ";
//		return 1;
//	}
//	fin.read((char*)&Proj[0], 300 * 40 * 3502 * sizeof(double));
//	thrust::host_vector<double> GF_2 = KatsevichFiltering_GPU<double>(Proj, 750, 106.7778696422639, 18.005134774901215, 75.0, 150.0, 12.5, 0.5, 2.461918834681550, 2, 300, 40, 3502);
//	std::ofstream fou("GF_Final_20181212.raw", std::ios::binary);
//	fou.write((char*)&GF_2[0], 300 * 40 * 3502 * sizeof(double));
//	fou.close();
//
//    return 0;
//}