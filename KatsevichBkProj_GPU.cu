/*
 ============================================================================
 Name        : KatsevichBkProj_GPU.cu
 Author      : Rui Liu
 Version     : 1.0
 Copyright   : Your copyright notice
 Description : The Katsevich Backprojection implementation in GPU
 ============================================================================
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include <mex.h>

#define TWOPI (6.283185307179586)
#define INV_TWOPI (0.1590250231624044)
#define PI (3.141592653589793)

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

#ifndef EPSILON
#define EPSILON (0.0000001)
#endif




//Create texture object and corresponding cudaArray function
template<typename T>
void createTextureObject(
	cudaTextureObject_t& texObj, //return: texture object pointing to the cudaArray
	cudaArray* d_prjArray, // return: cudaArray storing the data
	int Width, int Height, int Depth, // data size
	T* sourceData, // where is the data
	cudaMemcpyKind memcpyKind, // data from host or memory
	cudaTextureAddressMode addressMode, // how to address the texture (clamp, border ...)
	cudaTextureFilterMode textureFilterMode, // usually linear filtering (double --> int2 use pointer not linear interpolation)
	cudaTextureReadMode textureReadMode, // usually use element wise reading mode.
	bool isNormalized) // usually false
{
	cudaExtent prjSize;
	prjSize.width = Width;
	prjSize.height = Height;
	prjSize.depth = Depth;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
	cudaMalloc3DArray(&d_prjArray, &channelDesc, prjSize);
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(
		(void*)sourceData, prjSize.width * sizeof(T),
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
	CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
}


// Destroy a GPU array and corresponding TextureObject
void destroyTextureObject(cudaTextureObject_t& texObj, cudaArray* d_array)
{
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(d_array);
}




template<typename T>
__device__ __host__ inline void PISegment(const T x, const T y, const T z, T& BAngle, T& TAngle)
{
	T delta = 1.0;
	T bmax = z * TWOPI;
	T bmin = bmax - TWOPI;

	T r2 = x * x + y * y;

	T sb = 0;
	T st = 0;

	while ((bmax - bmin > EPSILON) && (delta > EPSILON))
	{
		sb = (bmax + bmin) * 0.5;
		T sinsb = sin(sb);
		T cossb = cos(sb);
		T tempcos = 2.0 * (1.0 - y * sinsb - x * cossb);
		assert(tempcos != 0);
		T t = (1.0 - r2) / tempcos;
		T templan = acos((y * cossb - x * sinsb) / sqrt(tempcos + r2 - 1.0));
		st = 2.0 * templan + sb;
		T zz = (sb * t + (1.0 - t) * st) * INV_TWOPI;
		if (zz < z)
		{
			bmin = sb;
		}
		else
		{
			bmax = sb;
		}
		delta = fabs(zz - z);
	}

	BAngle = sb;
	TAngle = st;
}


// Note: this projection do not consider the edging situation for backprojection
__global__ void backProjectionKer(
	float* Image,
	cudaTextureObject_t ProjTex,
	int RecMX, int RecMY, int RecMZ,
	float ObjRSquare,
	float* __restrict__ xCor, float* __restrict__ yCor, float* __restrict__ zCor,
	float ScanR,
	float StdDis,
	float DeltaU, float DeltaV,
	float HalfY, float HalfZ,
	float HelicP,
	float DeltaL,
	float ProjCtr,
	int ProjBeginIndex,
	int ProjNum,
	int ProjScale)
{
	int Zindex = threadIdx.x + blockIdx.x * blockDim.x;
	int Xindex = threadIdx.y + blockIdx.y * blockDim.y;
	int Yindex = threadIdx.z + blockIdx.z * blockDim.z;
	const size_t idx = (Xindex * RecMY + Yindex) * RecMZ + Zindex;
	if (Xindex < RecMX && Yindex < RecMY && Zindex < RecMZ)
	{
		const float X = xCor[Xindex];
		const float Y = yCor[Yindex];
		const float Z = zCor[Zindex];
		if (pow(X, 2.0f) + pow(Y, 2.0f) >= ObjRSquare)
			return;

		double BAngle;
		double TAngle;
		PISegment<double>(X / ScanR, Y / ScanR, Z / HelicP, BAngle, TAngle);
		BAngle = BAngle / DeltaL + ProjCtr - ProjBeginIndex;
		TAngle = TAngle / DeltaL + ProjCtr - ProjBeginIndex;

		int Bindex = int(BAngle);
		int Tindex = ceil(TAngle);
		if (Bindex < 0)
		{
			Bindex = 0;
		}
		if (Bindex > ProjNum - 1)
		{
			Bindex = ProjNum - 1;
		}
		if (Tindex < 0)
		{
			Tindex = 0;
		}
		if (Tindex > ProjNum - 1)
		{
			Tindex = ProjNum - 1;
		}
		float tpdata = 0.0f;
		for (int ProjIndex = Bindex; ProjIndex <= Tindex; ProjIndex++)
		{
			float theta = (ProjIndex + ProjBeginIndex - ProjCtr) * DeltaL;
			float cost = cosf(theta);
			float sint = sinf(theta);

			float DPSx = X - ScanR * cost;
			float DPSy = Y - ScanR * sint;
			float DPSz = Z - HelicP * theta * INV_TWOPI;
			float factor = sqrtf(DPSx * DPSx + DPSy * DPSy + DPSz * DPSz);
			float fenmu = -(DPSx * cost + DPSy * sint);
			float YY = DPSy * cost - DPSx * sint;
			YY = YY * StdDis / (fenmu * DeltaU) + HalfY;
			float ZZ = DPSz * StdDis / (fenmu * DeltaV) + HalfZ;
			float temp = tex3D<float>(ProjTex, YY + 0.5f, ZZ + 0.5f, ProjIndex + 0.5f);
			tpdata += temp / factor;
		}
		tpdata = -tpdata / ProjScale;
		Image[idx] = tpdata;

	}
}



void backProjection_internal(
	thrust::host_vector<float>& hImage,
	thrust::host_vector<float>& hProj,
	int RecMX, int RecMY, int RecMZ,
	float ObjRSquare,
	const thrust::host_vector<float>& hxCor,
	const thrust::host_vector<float>& hyCor,
	const thrust::host_vector<float>& hzCor,
	float ScanR,
	float StdDis,
	float DeltaU, float DeltaV,
	float HalfY, float HalfZ,
	int YL, int YLZL,
	float HelicP,
	float DeltaL,
	float ProjCtr,
	float ProjBeginIndex,
	int ProjNum, // number of projections
	int ProjScale,
	int threadidx, int threadidy, int threadidz)
{
	thrust::device_vector<float> Image = hImage;
	thrust::device_vector<float> xCor = hxCor;
	thrust::device_vector<float> yCor = hyCor;
	thrust::device_vector<float> zCor = hzCor;

	dim3 blk(threadidx, threadidy, threadidz);
	dim3 gid(
		(RecMZ + blk.x - 1) / blk.x,
		(RecMX + blk.y - 1) / blk.y,
		(RecMY + blk.z - 1) / blk.z);
	int ZL = YLZL / YL;
	cudaTextureObject_t projTex;
	cudaArray* d_projArray = nullptr;
	createTextureObject<float>(projTex, d_projArray,
		YL, ZL, ProjNum,
		&(hProj[0]),
		cudaMemcpyHostToDevice,
		cudaAddressModeClamp,
		cudaFilterModeLinear,
		cudaReadModeElementType, false);

	backProjectionKer<< <gid, blk >> > (
		thrust::raw_pointer_cast(&Image[0]),
		projTex,
		RecMX, RecMY, RecMZ, ObjRSquare,
		thrust::raw_pointer_cast(&xCor[0]),
		thrust::raw_pointer_cast(&yCor[0]),
		thrust::raw_pointer_cast(&zCor[0]),
		ScanR, StdDis, DeltaU, DeltaV, HalfY, HalfZ,
		HelicP, DeltaL, ProjCtr, ProjBeginIndex,
		ProjNum, ProjScale);
	hImage = Image;
	destroyTextureObject(projTex, d_projArray);
}


void backProjection(
	double* hImage,const double* hProj,
	int RecMX, int RecMY, int RecMZ,
	double ObjRSquare,
	double* hxCor, double* hyCor, double* hzCor,
	double ScanR,
	double StdDis,
	double DeltaU, double DeltaV,
	double HalfY, double HalfZ,
	int YL, int YLZL,
	double HelicP, double DeltaL, double ProjCtr,
	int ProjBeginIndex,
	int ProjNum, // number of projections
	int ProjScale,
	int threadidx, int threadidy, int threadidz)
{
	thrust::host_vector<float> Image(RecMX * RecMY * RecMZ, 0.0f);
	thrust::host_vector<float> Proj(ProjNum * YLZL, 0.0f);
	thrust::host_vector<float> xCor(ProjNum, 0.0f);
	thrust::host_vector<float> yCor(ProjNum, 0.0f);
	thrust::host_vector<float> zCor(ProjNum, 0.0f);

	thrust::fill(Image.begin(), Image.end(), 0.0f);

	thrust::copy(hProj, hProj + ProjNum * YLZL, &(Proj[0]));
	thrust::copy(hxCor, hxCor + RecMX, &(xCor[0]));
	thrust::copy(hyCor, hyCor + RecMY, &(yCor[0]));
	thrust::copy(hzCor, hzCor + RecMZ, &(zCor[0]));

	backProjection_internal(Image, Proj, RecMX, RecMY, RecMZ, ObjRSquare,
		xCor, yCor, zCor,
		ScanR, StdDis, DeltaU, DeltaV, HalfY, HalfZ, YL, YLZL,
		HelicP, DeltaL, ProjCtr, ProjBeginIndex, ProjNum,
		ProjScale, threadidx, threadidy, threadidz);
	thrust::copy(&(Image[0]), &(Image[0]) + RecMX * RecMY * RecMZ, hImage);
}


struct ReconData
{
	double* image;
	const double* proj;
	double* work_space;
};


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{

	const double *geom[3];

	// Check the correctness of the number of plhs and prhs.

	// Arguments about scanning locus: ScanR, StdDis, HelicP, ProjScale, ProjBeginIndex, ProjEndIndex
	geom[0] = mxGetPr(mxGetFieldByNumber(prhs[0], 0, 0));

	// Arguments about detector: DecWidth, YL, DecHeigh, ZL
	geom[1] = mxGetPr(mxGetFieldByNumber(prhs[0], 0, 1));

	// Arguments about object: ObjR, RecMx, RecMy, RecMz
	geom[2] = mxGetPr(mxGetFieldByNumber(prhs[0], 0, 2));

	ReconData data;
	data.proj = mxGetPr(prhs[1]);
	data.image = mxGetPr(prhs[2]);

	// system parameter
	double ScanR = geom[0][0];
	double StdDis = geom[0][1];
	double HelicP = geom[0][2];
	int ProjScale = int(geom[0][3]);
	int ProjBeginIndex = int(geom[0][4]);
	int ProjEndIndex = int(geom[0][5]);
	int ProjNum = ProjEndIndex - ProjBeginIndex + 1;
	double ProjCtr = geom[0][6];

	double DecWidth = geom[1][0];
	int YL = int(geom[1][1]);
	double DecHeigh = geom[1][2];
	int ZL = int(geom[1][3]);
	double DetCtrWidth = geom[1][4];
	double DetCtrHeigh = geom[1][5];



	double ObjR = geom[2][0];
	int RecMx = int(geom[2][1]);
	int RecMy = int(geom[2][2]);
	int RecMz = int(geom[2][3]);

	 //mexPrintf("hello\n");


	//Compute some often used constant*/
	double DeltaL = 2.0 * PI / ProjScale;
	double DeltaU = DecWidth / YL;
	double DeltaV = DecHeigh / ZL;
	double HalfY  = DetCtrWidth;//YL - 1.0) * 0.5;
	double HalfZ  = DetCtrHeigh; //(ZL - 1.0) * 0.5;
	double DeltaX = 2.0 * ObjR / RecMx;
	double DeltaY = 2.0 * ObjR / RecMy;
	double DeltaZ = 2.0 * ObjR / RecMz;
	double XCenter = (RecMx - 1.0) * 0.5;
	double YCenter = (RecMy - 1.0) * 0.5;
	double ZCenter = (RecMz - 1.0) * 0.5;

    int YLZL   = YL*ZL;

    double* Image  = data.image;
    const double* Proj   = data.proj;
    double ObjRSquare = ObjR * ObjR;

    double* xCor = new double[RecMx];
    double* yCor = new double[RecMy];
    double* zCor = new double[RecMz];

	for (int loop = 0; loop < RecMx; loop++)
	{
		xCor[loop] = (loop - XCenter) * DeltaX;
	}
	for (int loop = 0; loop < RecMy; loop++)
	{
		yCor[loop] = (loop - YCenter) * DeltaY;
	}
	for (int loop = 0; loop < RecMz; loop++)
	{
		zCor[loop] = (loop - ZCenter) * DeltaZ;
	}


	int threadidx = 32;
	int threadidy = 4;
	int threadidz = 4;

	backProjection(
		Image, Proj, RecMx, RecMy, RecMz,
		ObjRSquare, xCor, yCor, zCor, ScanR, StdDis, DeltaU, DeltaV,
		HalfY, HalfZ, YL, YLZL, HelicP, DeltaL, ProjCtr, ProjBeginIndex,
		ProjNum, ProjScale, threadidx, threadidy, threadidz);

	delete[] xCor;
	delete[] yCor;
	delete[] zCor;
}
//struct Locus
//{
//	double ScanR;
//	double StdDis;
//	double HelicP;
//	int ProjScale;
//	int ProjBeginIndex;
//	int ProjEndIndex;
//	double ProjCtr; // Center index of the projection data
//};
//
//struct Detector
//{
//	double DecWidth;
//	int YL;
//	double DecHeigh;
//	int ZL;
//};
//
//struct Object
//{
//	double ObjR;
//	int RecMX;
//	int RecMY;
//	int RecMZ;
//};
//
//
//
//
//int main()
//{
//	Locus locus;
//	Detector detector;
//	Object object;
//	locus.ScanR = 75.0f;
//	locus.StdDis = 150.0f;
//	locus.HelicP = 12.5f;
//	locus.ProjScale = 750;
//	locus.ProjBeginIndex = 1;
//	locus.ProjEndIndex = 3502;
//	locus.ProjCtr = 1751.0f;
//	detector.DecWidth = 106.7779;
//	detector.YL = 300;
//	detector.DecHeigh = 18.0051;
//	detector.ZL = 40;
//	object.ObjR = 25.0f;
//	object.RecMX = 1024;
//	object.RecMY = 1024;
//	object.RecMZ = 1024;
//	std::cout << object.RecMX * object.RecMY * object.RecMZ;
//
//	double DeltaL = TWOPI / static_cast<double>(locus.ProjScale);
//	double DeltaU = detector.DecWidth / static_cast<double>(detector.YL);
//	double DeltaV = detector.DecHeigh / static_cast<double>(detector.ZL);
//	double HalfY = (detector.YL - 1.0) * 0.5;
//	double HalfZ = (detector.ZL - 1.0) * 0.5;
//	double DeltaX = 2.0 * object.ObjR / static_cast<double>(object.RecMX);
//	double DeltaY = 2.0 * object.ObjR / static_cast<double>(object.RecMY);
//	double DeltaZ = 2.0 * object.ObjR / static_cast<double>(object.RecMZ);
//
//	double XCenter = (object.RecMX - 1.0) * 0.5;
//	double YCenter = (object.RecMY - 1.0) * 0.5;
//	double ZCenter = (object.RecMZ - 1.0) * 0.5;
//
//	int ProjNum = locus.ProjEndIndex - locus.ProjBeginIndex + 1;
//
//	double* VectorS = new double[ProjNum * 3];
//	double* VectorE1 = new double[ProjNum * 2];
//	double* VectorE2 = new double[ProjNum * 2];
//	double* xCor = new double[object.RecMX];
//	double* yCor = new double[object.RecMY];
//	double* zCor = new double[object.RecMZ];
//
//
//	for (int loop = 0; loop<ProjNum; loop++)
//	{
//		double temp = (loop + locus.ProjBeginIndex - locus.ProjCtr) * DeltaL;
//		VectorS[loop * 3] = locus.ScanR * cos(temp);
//		VectorS[loop * 3 + 1] = locus.ScanR * sin(temp);
//		VectorS[loop * 3 + 2] = locus.HelicP * temp * INV_TWOPI;
//		VectorE1[loop * 2] = -cos(temp);
//		VectorE1[loop * 2 + 1] = -sin(temp);
//		VectorE2[loop * 2] = -sin(temp);
//		VectorE2[loop * 2 + 1] = cos(temp);
//	}
//
//	for (int loop = 0; loop < object.RecMX; loop++)
//	{
//		xCor[loop] = (loop - XCenter) * DeltaX;
//	}
//	for (int loop = 0; loop < object.RecMY; loop++)
//	{
//		yCor[loop] = (loop - YCenter) * DeltaY;
//	}
//	for (int loop = 0; loop < object.RecMZ; loop++)
//	{
//		zCor[loop] = (loop - ZCenter) * DeltaZ;
//	}
//
//	int YLZL = detector.YL * detector.ZL;
//	double ObjRSquare = object.ObjR * object.ObjR;
//
//	double* hImage = new double[object.RecMX * object.RecMY * object.RecMZ];
//	double* hProj = new double[ProjNum * YLZL];
//	std::ifstream fin("/home/ruiliu/Desktop/katsevich/filteredProjection.prj", std::ios::binary);
//	if (!fin.is_open())
//	{
//		std::cout << "Cannot open the file\n";
//		return 1;
//	}
//	fin.read((char*)hProj, sizeof(double) * ProjNum * YLZL);
//	fin.close();
//
//	backProjection(
//		hImage, hProj,
//		object.RecMX, object.RecMY, object.RecMZ, ObjRSquare, xCor, yCor, zCor,
//		locus.ScanR, locus.StdDis, DeltaU, DeltaV, HalfY, HalfZ, detector.YL, YLZL,
//		locus.HelicP, DeltaL, locus.ProjCtr, locus.ProjBeginIndex, ProjNum, locus.ProjScale, 16, 4, 4);
//	std::ofstream fou("Backprojected.raw", std::ios::binary);
//	fou.write((char*)hImage, sizeof(double) * object.RecMX * object.RecMY * object.RecMZ);
//	fou.close();
//
//    return 0;
//}
//

