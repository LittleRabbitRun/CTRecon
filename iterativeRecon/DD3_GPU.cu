/**
* The MathWorks, Inc
* GPU based projectors.
* Four methods are implemented : (1) Branchless DD, (2) Double precision branchless DD
* (3) pseudo DD and (4) volume rendering method.
* author: Rui Liu
* date: 2019.01.08
* version: 2.0
*/

#include "mex.h"
#include "matrix.h"
#include <string>
#include <iostream>
#include "omp.h"
#include "utilities.cuh"

typedef unsigned char byte;

#define BLKX 32
#define BLKY 8
#define BLKZ 1

#define BACK_BLKX 64
#define BACK_BLKY 4
#define BACK_BLKZ 1

// Back projection methods
enum BackProjectionMethod { _BRANCHLESS, _PSEUDODD, _ZLINEBRANCHLESS, _VOLUMERENDERING };



// \brief Copy One Volume to Two, they are aligned in ZXY and ZYX orders
// separately and with one side padding with 0.
template<typename Ta, typename Tb>
__global__ void naive_copyToTwoVolumes(Ta*  in_ZXY, Tb* out_ZXY, Tb* out_ZYX, int XN, int YN, int ZN)
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


// Generate SAT of the volume along herizontal XY plane direction
template<typename Ta, typename Tb>
__global__ void naive_herizontalIntegral(Ta*  in, Tb* out, int N, int ZN)
{
    int zi = threadIdx.x + blockIdx.x * blockDim.x;
    if (zi < ZN)
    {
        out[zi] = in[zi];
        for (int i = 1;i<N;++i)
        {
            out[i * ZN + zi] = out[(i - 1) * ZN + zi] + in[i * ZN + zi];
        }
    }
}
// Generate SAT of the volume along vertical Z direction
template<typename Ta, typename Tb>
__global__ void naive_verticalIntegral(Ta*  in, Tb* out, int N, int ZN)
{
    int xyi = threadIdx.x + blockIdx.x * blockDim.x;
    if (xyi < N)
    {
        out[xyi * ZN] = in[xyi * ZN];
        for (int ii = 1; ii < ZN; ++ii)
        {
            out[xyi * ZN + ii] = out[xyi * ZN + ii - 1] + in[xyi * ZN + ii];
        }
    }
}

// template specialization for double precision into int2 datatype
template<>
__global__ void naive_verticalIntegral(double*  in, int2* out, int N, int ZN)
{
    int xyi = threadIdx.x + blockIdx.x * blockDim.x;
    if (xyi < N)
    {
        double temp = in[xyi * ZN];
        out[xyi * ZN] = make_int2(__double2loint(temp), __double2hiint(temp));
        double temp2 = 0;
        for (int ii = 1; ii < ZN; ++ii)
        {
            temp2 = temp + in[xyi * ZN + ii];
            out[xyi * ZN + ii] = make_int2(__double2loint(temp2), __double2hiint(temp2));
            temp = temp2;
        }
    }
}

// Generate SAT inplace without extra memory space in vertical Z direction
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

// Generate SAT inplace without extra memory space in horizontal Z direction
__global__ void horizontalIntegral(float* prj, int DNU, int DNV, int PN)
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


// Input the volume data in host memory, generate the SAT in ZXY storing order and ZYX storing order
// loaded in device memory
void genSAT_Of_Volume(float* hvol,
    thrust::device_vector<float>& ZXY,
    thrust::device_vector<float>& ZYX,
    int XN, int YN, int ZN)
{

    const int siz = XN * YN * ZN;
    const int nsiz_ZXY = (ZN + 1) * (XN + 1) * YN;
    const int nsiz_ZYX = (ZN + 1) * (YN + 1) * XN;
    ZXY.resize(nsiz_ZXY);
    ZYX.resize(nsiz_ZYX);
    thrust::device_vector<float> vol(hvol, hvol + siz);

    dim3 blk(64, 16, 1);
    dim3 gid(
        (ZN + blk.x - 1) / blk.x,
        (XN + blk.y - 1) / blk.y,
        (YN + blk.z - 1) / blk.z);
    // copy to original and transposed image volume with left- and top-side
    // boarder padding to be consistent with SAT dimensions
    naive_copyToTwoVolumes << <gid, blk >> >(thrust::raw_pointer_cast(&vol[0]),
        thrust::raw_pointer_cast(&ZXY[0]),
        thrust::raw_pointer_cast(&ZYX[0]), XN, YN, ZN);
    vol.clear();

    const int nZN = ZN + 1;
    const int nXN = XN + 1;
    const int nYN = YN + 1;

    //Generate SAT inplace ZXY

    blk.x = 32;
    blk.y = 1;
    blk.z = 1;
    gid.x = (nXN * YN + blk.x - 1) / blk.x;
    gid.y = 1;
    gid.z = 1;
    verticalIntegral << <gid, blk >> >(thrust::raw_pointer_cast(&ZXY[0]), nZN, nXN * YN);

    blk.x = 64;
    blk.y = 16;
    blk.z = 1;
    gid.x = (nZN + blk.x - 1) / blk.x;
    gid.y = (YN + blk.y - 1) / blk.y;
    gid.z = 1;
    horizontalIntegral << <gid, blk >> >(thrust::raw_pointer_cast(&ZXY[0]), nXN, nZN, YN);

    //Generate SAT inplace ZYX
    blk.x = 32;
    blk.y = 1;
    blk.z = 1;
    gid.x = (nYN * XN + blk.x - 1) / blk.x;
    gid.y = 1;
    gid.z = 1;
    verticalIntegral << <gid, blk >> >(thrust::raw_pointer_cast(&ZYX[0]), nZN, nYN * XN);

    blk.x = 64;
    blk.y = 16;
    blk.z = 1;
    gid.x = (nZN + blk.x - 1) / blk.x;
    gid.y = (XN + blk.y - 1) / blk.y;
    gid.z = 1;
    horizontalIntegral << <gid, blk >> >(thrust::raw_pointer_cast(&ZYX[0]), nYN, nZN, XN);
}


//Create texture object and corresponding cudaArray function
template<typename T>
void createTextureObject2(
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
void destroyTextureObject2(cudaTextureObject_t& texObj, cudaArray* d_array)
{
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(d_array);
}




// \brief Kernel function of Double precision based branchless DD with 2D SAT
__global__ void DD3_gpu_proj_doubleprecisionbranchless_ker(
    cudaTextureObject_t volTex1, // volume SAT in ZXY order
    cudaTextureObject_t volTex2, // volume SAT in ZYX order
    double* proj, // projection data
    double3 s, // initial source position
    const double3* __restrict cossinZT, // bind (cosine, sine, zshift)
    const double* __restrict xds,  //
    const double* __restrict yds,  //
    const double* __restrict zds,  // detector cells center positions
    const double* __restrict bxds,
    const double* __restrict byds,
    const double* __restrict bzds, // detector boundary positions
    double3 objCntIdx, // object center index
    double dx, double dz, // pixel size in xy plane and Z direction
    int XN, int YN, // pixel # in XY plane
    int DNU, int DNV, // detector cell # in xy plane and Z direction
    int PN) // view #
{
    int detIdV = threadIdx.x + blockIdx.x * blockDim.x;
    int detIdU = threadIdx.y + blockIdx.y * blockDim.y;
    int angIdx = threadIdx.z + blockIdx.z * blockDim.z;

    __shared__ double _xds[BLKY];
    __shared__ double _yds[BLKY];
    _xds[threadIdx.y] = xds[detIdU];
    _yds[threadIdx.y] = yds[detIdU];
    __syncthreads();

    if (detIdU < DNU && detIdV < DNV && angIdx < PN)
    {
        double3 dir = cossinZT[angIdx];
        double3 cursour = make_double3(
            s.x * dir.x - s.y * dir.y,
            s.x * dir.y + s.y * dir.x,
            s.z + dir.z);
        s = cossinZT[angIdx];
        double summ = _xds[threadIdx.y] * s.x - _yds[threadIdx.y] * s.y;
        double obj = _xds[threadIdx.y] * s.y + _yds[threadIdx.y] * s.x;
        double realL = bxds[detIdU];
        double realR = byds[detIdU];
        double realU = bxds[detIdU + 1];
        double realD = byds[detIdU + 1]; // intersection coordinates (mm); float2 is equv to (obj1,obj2) above
        double2 curDetL = make_double2(
            realL * s.x - realR * s.y,
            realL * s.y + realR * s.x);

        double2 curDetR = make_double2(
            realU * s.x - realD * s.y,
            realU * s.y + realD * s.x);
        double4 curDet = make_double4(summ, obj, bzds[detIdV] + s.z, bzds[detIdV + 1] + s.z); //(center x, center y, lower z, upper z)

        dir = normalize(make_double3(
            summ,
            obj,
            zds[detIdV] + s.z) - cursour);

        summ = 0; // to accumulate projection value
        obj = 0; // slice location (mm) along the ray tracing direction TODO: is this variable needed?

        double intersectLength, intersectHeight;
        double invdz = 1.0 / dz;
        double invdx = 1.0 / dx;


        double factL(1.0f); // dy/dx for (0,pi/4)
        double factR(1.0f);
        double factU(1.0f);
        double factD(1.0f);
        double constVal = 0;

        int crealD, crealR, crealU, crealL;
        int frealD, frealR, frealU, frealL;

        if (abs(s.x) <= abs(s.y))
        {
            summ = 0;
            // a few book keeping variables
            factL = (curDetL.y - cursour.y) / (curDetL.x - cursour.x);
            factR = (curDetR.y - cursour.y) / (curDetR.x - cursour.x);
            factU = (curDet.w - cursour.z) / (curDet.x - cursour.x);
            factD = (curDet.z - cursour.z) / (curDet.x - cursour.x);

            constVal = dx * dx * dz / (abs(dir.x));
#pragma unroll
            for (int ii = 0; ii < XN; ii++)
            {
                obj = (ii - objCntIdx.x) * dx;

                realL = (obj - curDetL.x) * factL + curDetL.y;
                realR = (obj - curDetR.x) * factR + curDetR.y;
                realU = (obj - curDet.x) * factU + curDet.w;
                realD = (obj - curDet.x) * factD + curDet.z;

                intersectLength = realR - realL;
                intersectHeight = realU - realD;

                // 1D LUT to address inaccuracies in texture coordinates
                realD = realD * invdz + objCntIdx.z + 1;
                realR = realR * invdx + objCntIdx.y + 1;
                realU = realU * invdz + objCntIdx.z + 1;
                realL = realL * invdx + objCntIdx.y + 1;

                crealD = ceil(realD);
                crealR = ceil(realR);
                crealU = ceil(realU);
                crealL = ceil(realL);

                frealD = floor(realD);
                frealR = floor(realR);
                frealU = floor(realU);
                frealL = floor(realL);


                summ +=
                    (bilerp(
                        tex3D<int2>(volTex2, frealD, frealL, ii + 0.5),
                        tex3D<int2>(volTex2, frealD, crealL, ii + 0.5),
                        tex3D<int2>(volTex2, crealD, frealL, ii + 0.5),
                        tex3D<int2>(volTex2, crealD, crealL, ii + 0.5),
                        realL - frealL, realD - frealD) +
                        bilerp(
                            tex3D<int2>(volTex2, frealU, frealR, ii + 0.5),
                            tex3D<int2>(volTex2, frealU, crealR, ii + 0.5),
                            tex3D<int2>(volTex2, crealU, frealR, ii + 0.5),
                            tex3D<int2>(volTex2, crealU, crealR, ii + 0.5),
                            realR - frealR, realU - frealU) -
                        bilerp(
                            tex3D<int2>(volTex2, frealD, frealR, ii + 0.5),
                            tex3D<int2>(volTex2, frealD, crealR, ii + 0.5),
                            tex3D<int2>(volTex2, crealD, frealR, ii + 0.5),
                            tex3D<int2>(volTex2, crealD, crealR, ii + 0.5),
                            realR - frealR, realD - frealD) -
                        bilerp(
                            tex3D<int2>(volTex2, frealU, frealL, ii + 0.5),
                            tex3D<int2>(volTex2, frealU, crealL, ii + 0.5),
                            tex3D<int2>(volTex2, crealU, frealL, ii + 0.5),
                            tex3D<int2>(volTex2, crealU, crealL, ii + 0.5),
                            realL - frealL, realU - frealU)) / (intersectLength * intersectHeight);
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
            constVal = dx * dx * dz / (abs(dir.y));
#pragma unroll
            for (int jj = 0; jj < YN; jj++)
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


                crealD = ceil(realD);
                crealR = ceil(realR);
                crealU = ceil(realU);
                crealL = ceil(realL);

                frealD = floor(realD);
                frealR = floor(realR);
                frealU = floor(realU);
                frealL = floor(realL);

                summ += (bilerp(
                    tex3D<int2>(volTex1, frealD, frealL, jj + 0.5),
                    tex3D<int2>(volTex1, frealD, crealL, jj + 0.5),
                    tex3D<int2>(volTex1, crealD, frealL, jj + 0.5),
                    tex3D<int2>(volTex1, crealD, crealL, jj + 0.5),
                    realL - frealL, realD - frealD) +
                    bilerp(
                        tex3D<int2>(volTex1, frealU, frealR, jj + 0.5),
                        tex3D<int2>(volTex1, frealU, crealR, jj + 0.5),
                        tex3D<int2>(volTex1, crealU, frealR, jj + 0.5),
                        tex3D<int2>(volTex1, crealU, crealR, jj + 0.5),
                        realR - frealR, realU - frealU) -
                    bilerp(
                        tex3D<int2>(volTex1, frealD, frealR, jj + 0.5),
                        tex3D<int2>(volTex1, frealD, crealR, jj + 0.5),
                        tex3D<int2>(volTex1, crealD, frealR, jj + 0.5),
                        tex3D<int2>(volTex1, crealD, crealR, jj + 0.5),
                        realR - frealR, realD - frealD) -
                    bilerp(
                        tex3D<int2>(volTex1, frealU, frealL, jj + 0.5),
                        tex3D<int2>(volTex1, frealU, crealL, jj + 0.5),
                        tex3D<int2>(volTex1, crealU, frealL, jj + 0.5),
                        tex3D<int2>(volTex1, crealU, crealL, jj + 0.5),
                        realL - frealL, realU - frealU)) / (intersectLength * intersectHeight);

            }
            __syncthreads();
            proj[(angIdx * DNU + detIdU) * DNV + detIdV] = summ * constVal;
        }

    }
}

// \brief C interface of Double Precision Branchless DD with 2D SAT
void DD3_gpu_proj_doubleprecisionbranchless(
    float x0, float y0, float z0,
    int DNU, int DNV,
    float* xds, float* yds, float* zds,
    float imgXCenter, float imgYCenter, float imgZCenter,
    float* hangs, float* hzPos, int PN,
    int XN, int YN, int ZN,
    float* vol, float* hprj,
    float dx, float dz,
    byte* mask, int gpunum)
{
    //Pre compute mask.*vol;
    for (int ii = 0; ii != XN * YN; ++ii)
    {
        byte v = mask[ii];
        for (int jj = 0; jj != ZN; ++jj)
        {
            vol[ii * ZN + jj] = vol[ii * ZN + jj] * v;
        }
    }

    float* bxds = new float[DNU + 1];
    float* byds = new float[DNU + 1];
    float* bzds = new float[DNV + 1];
    DD3Boundaries(DNU + 1, xds, bxds);
    DD3Boundaries(DNU + 1, yds, byds);
    DD3Boundaries(DNV + 1, zds, bzds);

    CUDA_SAFE_CALL(cudaSetDevice(gpunum));
    CUDA_SAFE_CALL(cudaDeviceReset());

    cudaStream_t streams[4];
    CUDA_SAFE_CALL(cudaStreamCreate(&streams[0]));
    CUDA_SAFE_CALL(cudaStreamCreate(&streams[1]));
    CUDA_SAFE_CALL(cudaStreamCreate(&streams[2]));
    CUDA_SAFE_CALL(cudaStreamCreate(&streams[3]));

    int TOTVN = XN * YN * ZN;
    double objCntIdxX = (XN - 1.0) * 0.5 - imgXCenter / dx;
    double objCntIdxY = (YN - 1.0) * 0.5 - imgYCenter / dx;
    double objCntIdxZ = (ZN - 1.0) * 0.5 - imgZCenter / dz;

    thrust::device_vector<float> in(vol, vol + TOTVN); // original img volume
    thrust::device_vector<double> in_ZXY((ZN + 1) * (XN + 1) * YN, 0); //
    thrust::device_vector<double> in_ZYX((ZN + 1) * (YN + 1) * XN, 0); // transposed img volume

    dim3 blk(64, 16, 1);
    dim3 gid(
        (ZN + blk.x - 1) / blk.x,
        (XN + blk.y - 1) / blk.y,
        (YN + blk.z - 1) / blk.z);
    // copy to original and transposed image volume with left- and top-side boarder padding to be consistent with SAT dimensions
    naive_copyToTwoVolumes<float, double> << <gid, blk >> >(
        thrust::raw_pointer_cast(&in[0]),
        thrust::raw_pointer_cast(&in_ZXY[0]),
        thrust::raw_pointer_cast(&in_ZYX[0]), XN, YN, ZN);
    in.clear();

    thrust::device_vector<double> in_ZXY_summ1((ZN + 1) * (XN + 1) * YN, 0);
    thrust::device_vector<int2> in_ZXY_summ((ZN + 1) * (XN + 1) * YN);


    blk.x = 64;							blk.y = 1;		blk.z = 1;
    gid.x = (ZN + blk.x) / blk.x;		gid.y = 1;		gid.z = 1;

    dim3 blk2(64);
    dim3 gid2((YN + blk2.x) / blk2.x);
    dim3 blk3(64);
    dim3 gid3((XN + blk3.x) / blk3.x);

    // compute SAT for the original img volume
    for (int jj = 0; jj != YN; ++jj)
    {
        // for each Y slice
        naive_herizontalIntegral << <gid, blk, 0, streams[0] >> >(
            thrust::raw_pointer_cast(&in_ZXY[0]) + jj * (ZN + 1) * (XN + 1),
            thrust::raw_pointer_cast(&in_ZXY_summ1[0]) + jj * (ZN + 1) * (XN + 1), XN + 1, ZN + 1);
        naive_verticalIntegral << <gid2, blk2, 0, streams[0] >> >(
            thrust::raw_pointer_cast(&in_ZXY_summ1[0]) + jj * (ZN + 1) * (XN + 1),
            thrust::raw_pointer_cast(&in_ZXY_summ[0]) + jj * (ZN + 1) * (XN + 1), XN + 1, ZN + 1);
    }
    in_ZXY.clear();
    in_ZXY_summ1.clear();


    cudaArray* d_volumeArray1 = nullptr;
    cudaTextureObject_t texObj1;

    createTextureObject2<int2>(texObj1, d_volumeArray1, ZN + 1, XN + 1, YN,
        thrust::raw_pointer_cast(&in_ZXY_summ[0]),
        cudaMemcpyDeviceToDevice, cudaAddressModeClamp, cudaFilterModePoint,
        cudaReadModeElementType, false);
    in_ZXY_summ.clear();






    thrust::device_vector<double> in_ZYX_summ1((ZN + 1) * (YN + 1) * XN, 0); // SAT for the transposed img volume
    thrust::device_vector<int2> in_ZYX_summ((ZN + 1) * (YN + 1) * XN);
    // compute SAT for the transposed img volume
    for (int ii = 0; ii != XN; ++ii)
    {
        // for each X slice
        naive_herizontalIntegral << <gid, blk, 0, streams[1] >> >(
            thrust::raw_pointer_cast(&in_ZYX[0]) + ii * (ZN + 1) * (YN + 1),
            thrust::raw_pointer_cast(&in_ZYX_summ1[0]) + ii * (ZN + 1) * (YN + 1), YN + 1, ZN + 1);
        naive_verticalIntegral << <gid3, blk3, 0, streams[1] >> >(
            thrust::raw_pointer_cast(&in_ZYX_summ1[0]) + ii * (ZN + 1) * (YN + 1),
            thrust::raw_pointer_cast(&in_ZYX_summ[0]) + ii * (ZN + 1) * (YN + 1), YN + 1, ZN + 1);
    }
    in_ZYX.clear();
    in_ZYX_summ1.clear();

    cudaArray* d_volumeArray2 = nullptr;
    cudaTextureObject_t texObj2;

    createTextureObject2<int2>(texObj2, d_volumeArray2, ZN + 1, YN + 1, XN,
        thrust::raw_pointer_cast(&in_ZYX_summ[0]),
        cudaMemcpyDeviceToDevice, cudaAddressModeClamp, cudaFilterModePoint,
        cudaReadModeElementType, false);
    in_ZYX_summ.clear();

    thrust::device_vector<double> prj(DNU * DNV * PN, 0);
    thrust::device_vector<double> angs(hangs, hangs + PN);
    thrust::device_vector<double> zPos(hzPos, hzPos + PN);
    thrust::device_vector<double> d_xds(xds, xds + DNU);
    thrust::device_vector<double> d_yds(yds, yds + DNU);
    thrust::device_vector<double> d_zds(zds, zds + DNV);
    thrust::device_vector<double> d_bxds(bxds, bxds + DNU + 1);
    thrust::device_vector<double> d_byds(byds, byds + DNU + 1);
    thrust::device_vector<double> d_bzds(bzds, bzds + DNV + 1);


    // constant values for DD calculation
    thrust::device_vector<double3> cossinZT(PN);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(angs.begin(), zPos.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(angs.end(), zPos.end())),
        cossinZT.begin(), CTMBIR::ConstantForBackProjection<double>(x0, y0, z0));

    //precalculate all constant values in CUDA
    dim3 blkc(64, 16, 1);
    dim3 gidc(
        (DNV + blkc.x) / blkc.x,
        (DNU + blkc.y) / blkc.y,
        (PN + blkc.z - 1) / blkc.z);


    //Configure BLOCKs for projection
    blk.x = BLKX; // det row index
    blk.y = BLKY; // det col index
    blk.z = BLKZ; // view index
    gid.x = (DNV + blk.x - 1) / blk.x;
    gid.y = (DNU + blk.y - 1) / blk.y;
    gid.z = (PN + blk.z - 1) / blk.z;

    //Projection kernel
    DD3_gpu_proj_doubleprecisionbranchless_ker << <gid, blk >> >(texObj1, texObj2,
        thrust::raw_pointer_cast(&prj[0]), make_double3(x0, y0, z0),
        thrust::raw_pointer_cast(&cossinZT[0]),
        thrust::raw_pointer_cast(&d_xds[0]),
        thrust::raw_pointer_cast(&d_yds[0]),
        thrust::raw_pointer_cast(&d_zds[0]),
        thrust::raw_pointer_cast(&d_bxds[0]),
        thrust::raw_pointer_cast(&d_byds[0]),
        thrust::raw_pointer_cast(&d_bzds[0]),
        make_double3(objCntIdxX, objCntIdxY, objCntIdxZ),
        dx, dz, XN, YN, DNU, DNV, PN);
    thrust::copy(prj.begin(), prj.end(), hprj);

    CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj1));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj2));

    destroyTextureObject2(texObj1, d_volumeArray1);
    destroyTextureObject2(texObj2, d_volumeArray2);
    prj.clear();
    angs.clear();
    zPos.clear();

    d_xds.clear();
    d_yds.clear();
    d_zds.clear();
    d_bxds.clear();
    d_byds.clear();
    d_bzds.clear();


    delete[] bxds;
    delete[] byds;
    delete[] bzds;
}












// \brief Kernel function of Pseudo Distance Driven without SAT
__global__ void DD3_gpu_proj_pseudodistancedriven_ker(
    cudaTextureObject_t volTex,
    float* proj, float3 s,
    float*  d_xds, float*  d_yds, float*  d_zds,
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

            for (int jj = 0; jj < YN; ++jj)
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

// \brief C interface of pseudo distance driven
void DD3_gpu_proj_pseudodistancedriven(
    float x0, float y0, float z0,
    int DNU, int DNV,
    float* xds, float* yds, float* zds,
    float imgXCenter, float imgYCenter, float imgZCenter,
    float* hangs, float* hzPos, int PN,
    int XN, int YN, int ZN, float* hvol, float* hprj,
    float dx, float dz, byte* mask, int gpunum)
{
    //Pre compute mask.*vol;
    for (int ii = 0; ii != XN * YN; ++ii)
    {
        byte v = mask[ii];
        for (int jj = 0; jj != ZN; ++jj)
        {
            hvol[ii * ZN + jj] = hvol[ii * ZN + jj] * v;
        }
    }

    float* bxds = new float[DNU + 1];
    float* byds = new float[DNU + 1];
    float* bzds = new float[DNV + 1];
    DD3Boundaries(DNU + 1, xds, bxds);
    DD3Boundaries(DNU + 1, yds, byds);
    DD3Boundaries(DNV + 1, zds, bzds);

    CUDA_SAFE_CALL(cudaSetDevice(gpunum));
    CUDA_SAFE_CALL(cudaDeviceReset());

    const int TOTVN = XN * YN * ZN;
    float objCntIdxX = (XN - 1.0) * 0.5 - imgXCenter / dx;
    float objCntIdxY = (YN - 1.0) * 0.5 - imgYCenter / dx;
    float objCntIdxZ = (ZN - 1.0) * 0.5 - imgZCenter / dz;

    d_vec_t vol(hvol, hvol + TOTVN);
    // Allocate CUDA array in device memory
    cudaTextureObject_t texObj;
    cudaArray* d_volumeArray = nullptr;
    createTextureObject2<float>(texObj, d_volumeArray,
        ZN, XN, YN,
        thrust::raw_pointer_cast(&vol[0]),
        cudaMemcpyDeviceToDevice,
        cudaAddressModeBorder,
        cudaFilterModeLinear,
        cudaReadModeElementType, false);

    //Calculate constant values.
    d_vec_t prj(DNU * DNV * PN, 0);
    d_vec_t angs(hangs, hangs + PN);
    d_vec_t zPos(hzPos, hzPos + PN);
    d_vec_t d_xds(xds, xds + DNU);
    d_vec_t d_yds(yds, yds + DNU);
    d_vec_t d_zds(zds, zds + DNV);

    thrust::device_vector<float3> cossinZT(PN);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(angs.begin(), zPos.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(angs.end(), zPos.end())),
        cossinZT.begin(), CTMBIR::ConstantForBackProjection<float>(x0, y0, z0));

    dim3 blk(32, 8, 1);
    dim3 gid((DNV + blk.x - 1) / blk.x,
        (DNU + blk.y - 1) / blk.y,
        (PN + blk.z - 1) / blk.z);

    DD3_gpu_proj_pseudodistancedriven_ker << <gid, blk >> >(texObj, thrust::raw_pointer_cast(&prj[0]),
        make_float3(x0, y0, z0),
        thrust::raw_pointer_cast(&d_xds[0]),
        thrust::raw_pointer_cast(&d_yds[0]),
        thrust::raw_pointer_cast(&d_zds[0]),
        thrust::raw_pointer_cast(&cossinZT[0]),
        make_float3(objCntIdxX, objCntIdxY, objCntIdxZ),
        dx, dz, XN, YN, DNU, DNV, PN);
    thrust::copy(prj.begin(), prj.end(), hprj);


    destroyTextureObject2(texObj, d_volumeArray);

    prj.clear();
    angs.clear();
    zPos.clear();
    d_xds.clear();
    d_yds.clear();
    d_zds.clear();
    cossinZT.clear();

    delete[] bxds;
    delete[] byds;
    delete[] bzds;
}









// \brief Kernel function for branchless DD with 2D SAT
__global__ void DD3_gpu_proj_branchless_sat2d_ker(
    cudaTextureObject_t volTex1,
    cudaTextureObject_t volTex2,
    float* proj,
    float3 s,
    const float3* __restrict cossinZT,
    const float* __restrict xds,
    const float* __restrict yds,
    const float* __restrict zds,
    const float* __restrict bxds,
    const float* __restrict byds,
    const float* __restrict bzds,
    float3 objCntIdx,
    float dx, float dz,
    int XN, int YN,
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
        float realD = byds[detIdU + 1]; // intersection coordinates (mm); float2 is equv to (obj1,obj2) above
        float2 curDetL = make_float2(
            realL * s.x - realR * s.y,
            realL * s.y + realR * s.x);

        float2 curDetR = make_float2(
            realU * s.x - realD * s.y,
            realU * s.y + realD * s.x);
        float4 curDet = make_float4(summ, obj, bzds[detIdV] + s.z, bzds[detIdV + 1] + s.z); //(center x, center y, lower z, upper z)

        dir = normalize(make_float3(
            summ,
            obj,
            zds[detIdV] + s.z) - cursour);

        summ = 0; // to accumulate projection value
        obj = 0; // slice location (mm) along the ray tracing direction

        float intersectLength, intersectHeight;
        float invdz = 1.0 / dz;
        float invdx = 1.0 / dx;


        float factL(1.0f); // dy/dx for (0,pi/4)
        float factR(1.0f);
        float factU(1.0f);
        float factD(1.0f);
        float constVal = 0;

        if (fabsf(s.x) <= fabsf(s.y))
        {
            summ = 0;
            // a few book keeping variables
            factL = (curDetL.y - cursour.y) / (curDetL.x - cursour.x);
            factR = (curDetR.y - cursour.y) / (curDetR.x - cursour.x);
            factU = (curDet.w - cursour.z) / (curDet.x - cursour.x);
            factD = (curDet.z - cursour.z) / (curDet.x - cursour.x);

            constVal = dx * dx * dz / (fabsf(dir.x));
#pragma unroll
            for (int ii = 0; ii < XN; ii++)
            {
                obj = (ii - objCntIdx.x) * dx;

                realL = (obj - curDetL.x) * factL + curDetL.y;
                realR = (obj - curDetR.x) * factR + curDetR.y;
                realU = (obj - curDet.x) * factU + curDet.w;
                realD = (obj - curDet.x) * factD + curDet.z;

                intersectLength = realR - realL;
                intersectHeight = realU - realD;

                // 1D LUT to address inaccuracies in texture coordinates
                realD = realD * invdz + objCntIdx.z + 1;
                realR = realR * invdx + objCntIdx.y + 1;
                realU = realU * invdz + objCntIdx.z + 1;
                realL = realL * invdx + objCntIdx.y + 1;

                summ += (tex3D<float>(volTex2, realD, realL, ii + 0.5f)
                    + tex3D<float>(volTex2, realU, realR, ii + 0.5f)
                    - (tex3D<float>(volTex2, realD, realR, ii + 0.5f)
                        + tex3D<float>(volTex2, realU, realL, ii + 0.5f))
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
            constVal = dx * dx * dz / (fabsf(dir.y));
#pragma unroll
            for (int jj = 0; jj < YN; jj++)
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

                summ += (tex3D<float>(volTex1, realD, realL, jj + 0.5f)
                    + tex3D<float>(volTex1, realU, realR, jj + 0.5f)
                    - (tex3D<float>(volTex1, realD, realR, jj + 0.5f) +
                        tex3D<float>(volTex1, realU, realL, jj + 0.5f))
                    ) / (intersectLength * intersectHeight);
            }
            __syncthreads();
            proj[(angIdx * DNU + detIdU) * DNV + detIdV] = summ * constVal;
        }

    }
}
// \brief C interface of branchless DD with 2D SAT
void DD3_gpu_proj_branchless_sat2d(
    float x0, float y0, float z0,
    int DNU, int DNV, // detector ch#, detector row#
    float* xds, float* yds, float* zds,
    float imgXCenter, float imgYCenter, float imgZCenter,
    float* hangs, float* hzPos, int PN,
    int XN, int YN, int ZN,
    float* vol, float* hprj,
    float dx, float dz,
    byte* mask, int gpunum)
{

    for (int i = 0; i != XN * YN; ++i)
    {
        byte v = mask[i];
        for (int z = 0; z != ZN; ++z)
        {
            vol[i * ZN + z] = vol[i * ZN + z] * v;
        }
    }
    CUDA_SAFE_CALL(cudaSetDevice(gpunum)); // choose the 2nd GPU for now (1st GPU used to be occupied on torb
    CUDA_SAFE_CALL(cudaDeviceReset());

    float* bxds = new float[DNU + 1];
    float* byds = new float[DNU + 1];
    float* bzds = new float[DNV + 1];

    DD3Boundaries(DNU + 1, xds, bxds);
    DD3Boundaries(DNU + 1, yds, byds);
    DD3Boundaries(DNV + 1, zds, bzds);

    cudaStream_t streams[4];
    CUDA_SAFE_CALL(cudaStreamCreate(&streams[0]));
    CUDA_SAFE_CALL(cudaStreamCreate(&streams[1]));
    CUDA_SAFE_CALL(cudaStreamCreate(&streams[2]));
    CUDA_SAFE_CALL(cudaStreamCreate(&streams[3]));

    float objCntIdxX = (XN - 1.0) * 0.5 - imgXCenter / dx;
    float objCntIdxY = (YN - 1.0) * 0.5 - imgYCenter / dx;
    float objCntIdxZ = (ZN - 1.0) * 0.5 - imgZCenter / dz;


    thrust::device_vector<float> SATZXY;
    thrust::device_vector<float> SATZYX;
    genSAT_Of_Volume(vol, SATZXY, SATZYX, XN, YN, ZN);


    /// Copy volumes to texture
    // Allocate CUDA array in device memory
    cudaTextureObject_t texObj1;
    cudaArray* d_volumeArray1 = nullptr;
    cudaTextureObject_t texObj2;
    cudaArray* d_volumeArray2 = nullptr;
    createTextureObject2<float>(texObj1, d_volumeArray1,
        ZN + 1, XN + 1, YN,
        thrust::raw_pointer_cast(&SATZXY[0]),
        cudaMemcpyDeviceToDevice,
        cudaAddressModeClamp,
        cudaFilterModeLinear,
        cudaReadModeElementType, false);
    SATZXY.clear();
    createTextureObject2<float>(texObj2, d_volumeArray2,
        ZN + 1, YN + 1, XN,
        thrust::raw_pointer_cast(&SATZYX[0]),
        cudaMemcpyDeviceToDevice,
        cudaAddressModeClamp,
        cudaFilterModeLinear,
        cudaReadModeElementType, false);
    SATZYX.clear();

    thrust::device_vector<float> prj(DNU * DNV * PN, 0); // projection data
    thrust::device_vector<float> d_xds(xds, xds + DNU); // detector positions (e.g., 888) in device mem
    thrust::device_vector<float> d_yds(yds, yds + DNU); // detector positions (e.g., 888) in device mem
    thrust::device_vector<float> d_zds(zds, zds + DNV); // detector positions (e.g., 888) in device mem
    thrust::device_vector<float> d_bxds(bxds, bxds + DNU + 1); // detector boundary positions (e.g., 889) in device mem
    thrust::device_vector<float> d_byds(byds, byds + DNU + 1); // detector boundary positions (e.g., 889) in device mem
    thrust::device_vector<float> d_bzds(bzds, bzds + DNV + 1); // detector boundary positions (e.g., 889) in device mem

                                                               // Allocate corresponding device memories

    thrust::device_vector<float> angs(hangs, hangs + PN);
    thrust::device_vector<float> zPos(hzPos, hzPos + PN);

    thrust::device_vector<float3> cossinZT(PN);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(angs.begin(), zPos.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(angs.end(), zPos.end())),
        cossinZT.begin(), CTMBIR::ConstantForBackProjection<float>(x0, y0, z0));

    //Configure BLOCKs for projection
    dim3 blk;
    dim3 gid;
    blk.x = BLKX; // det row index
    blk.y = BLKY; // det col index
    blk.z = BLKZ; // view index
    gid.x = (DNV + blk.x - 1) / blk.x;
    gid.y = (DNU + blk.y - 1) / blk.y;
    gid.z = (PN + blk.z - 1) / blk.z;

    //Projection kernel
    DD3_gpu_proj_branchless_sat2d_ker << <gid, blk >> >(texObj1, texObj2,
        thrust::raw_pointer_cast(&prj[0]),
        make_float3(x0, y0, z0),
        thrust::raw_pointer_cast(&cossinZT[0]),
        thrust::raw_pointer_cast(&d_xds[0]),
        thrust::raw_pointer_cast(&d_yds[0]),
        thrust::raw_pointer_cast(&d_zds[0]),
        thrust::raw_pointer_cast(&d_bxds[0]),
        thrust::raw_pointer_cast(&d_byds[0]),
        thrust::raw_pointer_cast(&d_bzds[0]),
        make_float3(objCntIdxX, objCntIdxY, objCntIdxZ), dx, dz, XN, YN, DNU, DNV, PN);
    thrust::copy(prj.begin(), prj.end(), hprj);

    CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj1));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj2));

    destroyTextureObject2(texObj1, d_volumeArray1);
    destroyTextureObject2(texObj2, d_volumeArray2);

    angs.clear();
    zPos.clear();
    prj.clear();
    d_xds.clear();
    d_yds.clear();
    d_zds.clear();
    d_bxds.clear();
    d_byds.clear();
    d_bzds.clear();
    cossinZT.clear();

    delete[] bxds;
    delete[] byds;
    delete[] bzds;

}





extern "C"
void DD3Proj_gpu(
    float x0, float y0, float z0, // src position in gantry coordinates
    int DNU, int DNV,  // detector channel#, detector row#
    float* xds, float* yds, float* zds, // detector center position in x, y, z dimension
    float imgXCenter, float imgYCenter, float imgZCenter, // img center in world coordinate
    float* hangs, float* hzPos, int PN, // view angles, src position in Z in world coordinate, view#
    int XN, int YN, int ZN, // dimension of image volume
    float* hvol, float* hprj, // img data, projection data
    float dx, float dz, // img size in xy, img size in z
    byte* mask, int gpunum, int prjMode)
{
    switch (prjMode)
    {
    case 0:
        DD3_gpu_proj_branchless_sat2d(x0, y0, z0, DNU, DNV,
            xds, yds, zds,
            imgXCenter, imgYCenter, imgZCenter,
            hangs, hzPos, PN, XN, YN, ZN,
            hvol, hprj, dx, dz, mask, gpunum);
        break;
    case 1:
        // disable school code
        DD3_gpu_proj_pseudodistancedriven(x0, y0, z0, DNU, DNV,
            xds, yds, zds,
            imgXCenter, imgYCenter, imgZCenter,
            hangs, hzPos, PN, XN, YN, ZN,
            hvol, hprj, dx, dz, mask, gpunum);
        break;
    case 2:
        DD3_gpu_proj_doubleprecisionbranchless(x0, y0, z0, DNU, DNV,
            xds, yds, zds,
            imgXCenter, imgYCenter, imgZCenter,
            hangs, hzPos, PN, XN, YN, ZN,
            hvol, hprj, dx, dz, mask, gpunum);
        break;
    case 3:
        DD3_gpu_proj_pseudodistancedriven(x0, y0, z0, DNU, DNV,
            xds, yds, zds,
            imgXCenter, imgYCenter, imgZCenter,
            hangs, hzPos, PN, XN, YN, ZN,
            hvol, hprj, dx, dz, mask, gpunum);
        break;
    default:
        DD3_gpu_proj_branchless_sat2d(x0, y0, z0, DNU, DNV,
            xds, yds, zds,
            imgXCenter, imgYCenter, imgZCenter,
            hangs, hzPos, PN, XN, YN, ZN,
            hvol, hprj, dx, dz, mask, gpunum);
        break;

    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef CALDETPARAS
#define CALDETPARAS
//Calculate the detCtrIdx and detector cell size parameters.
float4 calDetParas(float* xds, float* yds, float* zds, float x0, float y0, float z0, int DNU, int DNV)
{
    float* bxds = new float[DNU + 1];
    float* byds = new float[DNU + 1];
    float* bzds = new float[DNV + 1];

    DD3Boundaries(DNU + 1, xds, bxds);
    DD3Boundaries(DNU + 1, yds, byds);
    DD3Boundaries(DNV + 1, zds, bzds);

    // detector size in Z
    float ddv = (bzds[DNV] - bzds[0]) / DNV; // detector size in Z direction
    float detCtrIdxV = (-(bzds[0] - z0) / ddv) - 0.5; // detector center index in Z direction
    float2 dir = normalize(make_float2(-x0, -y0)); // source to origin vector (XY plane)
    float2 dirL = normalize(make_float2(bxds[0] - x0, byds[0] - y0)); // Left boundary direction vector
    float2 dirR = normalize(make_float2(bxds[DNU] - x0, byds[DNU] - y0)); // Right boundary direction vector
                                                                          // the angular size of detector cell as seen by the source
    float dbeta = asin(dirL.x * dirR.y - dirL.y * dirR.x) / DNU; //detector size in channel direction
    float minBeta = asin(dir.x * dirL.y - dir.y * dirL.x); //the fan angle corresponding to the most left boundary
    float detCtrIdxU = -minBeta / dbeta - 0.5; //det center index in XY / channel direction
    delete[] bxds;
    delete[] byds;
    delete[] bzds;
    return make_float4(detCtrIdxU, detCtrIdxV, dbeta, ddv);
}
#endif


// Copy the projection to a new space with edges padded with 0
// It adds an extra boarder line of zeros around a stack of 2D image (3D data)
// the image size will be increased by 2 pixels along both dimensions
// This is needed by pesudo-DD backprojection with non-uniform detector size (not tested)
__global__ void addTwoSidedZeroBoarder(float* prjIn, float* prjOut, const int DNU, const int DNV, const int PN)
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

// Copy the projection to a new space with left and upper edges padded with 0
// It adds an extra boarder line of zeros around a stack of 2D image (3D data) only on one side
// the image size will be increased by 1 pixels along both dimensions
// This is need by the summed area table used by branchless DD model
__global__ void addOneSidedZeroBoarder
(float* prj_in, float* prj_out, int DNU, int DNV, int PN)
{
    int idv = threadIdx.x + blockIdx.x * blockDim.x;
    int idu = threadIdx.y + blockIdx.y * blockDim.y;
    int ang = threadIdx.z + blockIdx.z * blockDim.z;
    if (idu < DNU && idv < DNV && ang < PN)
    {
        int i = (ang * DNU + idu) * DNV + idv;
        int ni = (ang * (DNU + 1) + (idu + 1)) * (DNV + 1) + (idv + 1);
        prj_out[ni] = prj_in[i];
    }
}


// The same as verticalIntegral in source file DD3_GPU_proj.cu to avoid redefinition error in nvcc compiler
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
// The same as horizontalIntegral in source file DD3_GPU_proj.cu to avoid redefinition error in nvcc compiler
__global__ void horizontalIntegral2(float* prj, int DNU, int DNV, int PN)
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
// \brief Generate SAT for projection
thrust::device_vector<float> genSAT_of_Projection(
    float* hprj, // Projection data in host memory
    int DNU, // Detector channel #
    int DNV, // Detector row #
    int PN)  // view #
{
    const int siz = DNU * DNV * PN;
    const int nsiz = (DNU + 1) * (DNV + 1) * PN;
    thrust::device_vector<float> prjSAT(nsiz, 0); // SAT in device memory
    thrust::device_vector<float> prj(hprj, hprj + siz); // projection in device memory
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

    //Generate SAT inplace
    copyBlk.x = 512;
    copyBlk.y = 1;
    copyBlk.z = 1;
    copyGid.x = (nDNU * PN + copyBlk.x - 1) / copyBlk.x;
    copyGid.y = 1;
    copyGid.z = 1;
    verticalIntegral2 << <copyGid, copyBlk >> >(thrust::raw_pointer_cast(&prjSAT[0]), nDNV, nDNU * PN);

    copyBlk.x = 64;
    copyBlk.y = 16;
    copyBlk.z = 1;
    copyGid.x = (nDNV + copyBlk.x - 1) / copyBlk.x;
    copyGid.y = (PN + copyBlk.y - 1) / copyBlk.y;
    copyGid.z = 1;
    horizontalIntegral2 << <copyGid, copyBlk >> >(thrust::raw_pointer_cast(&prjSAT[0]), nDNU, nDNV, PN);
    return prjSAT; // It has the device to device memory copy (TODO: avoid this by using reference parameters)
}


// Create a GPU array and corresponding TextureObject
template<typename T>
void createTextureObject(
    cudaTextureObject_t& texObj,
    cudaArray* d_prjArray, // Can be delete;
    int Width, int Height, int Depth,
    T* sourceData,
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



// Template of backprojection, the template parameter detemines which method is specialized and used.
template<BackProjectionMethod Method>
__global__ void DD3_gpu_back_ker(
    cudaTextureObject_t texObj, //projection texture object
    float* vol, // volume to be backprojected
    byte* __restrict__ msk, // backprojection mask (i.e. 512X512)
    float3* __restrict__ cossinZT, // bind (cosine, sine, zshift) into a float3 datatype
    float3 s, // initial source position
    float S2D, // source to detector distance
    float3 objCntIdx, // the index in the object where the origin locates
    float dx, float dz, // pixel size in XY plane and Z direction
    float dbeta, float ddv, // detector size in channel direction and Z direction
    float2 detCntIdx, // detector center index
    int3 VN, // volume pixel # in X,Y,Z directions
    int PN, // view #
    int squared) // it is useless now. (TODO: implement it)
{}


// Template specialization to Branchless DD backprojection
template<>
__global__ void DD3_gpu_back_ker<_BRANCHLESS>(
    cudaTextureObject_t texObj,
    float* vol,
    byte* __restrict__ msk,
    float3* __restrict__ cossinZT,
    float3 s,
    float S2D,
    float3 curvox,
    float dx, float dz, float dbeta, float ddv,
    float2 detCntIdx, int3 VN, int PN, int squared)
{
    int3 id;
    id.z = threadIdx.x + __umul24(blockIdx.x, blockDim.x);  // BACK_BLKX
    id.x = threadIdx.y + __umul24(blockIdx.y, blockDim.y);  // BACK_BLKY
    id.y = threadIdx.z + __umul24(blockIdx.z, blockDim.z);  // BACK_BLKZ

    if (id.x < VN.x && id.y < VN.y && id.z < VN.z)
    {
        if (msk[id.y * VN.x + id.x] != 1)
            return;
        // Position of current pixel
        curvox = (id - curvox) * make_float3(dx, dx, dz);// make_float3((id.x - objCntIdx.x) * dx, (id.y - objCntIdx.y) * dx, (id.z - objCntIdx.z) * dz);
        float3 cursour; // src position (precomputed in global mem "cursours"
        float idxL, idxR, idxU, idxD; // detctor index corresponding to shadow of the current pixel
        float cosVal; // ray angle relative to the normal vector of detector face
        float summ = 0;


        float3 cossin;
        float inv_sid = 1.0 / sqrtf(s.x * s.x + s.y * s.y);
        float3 dir;
        float l_square;
        float l;
        float alpha;
        //float t;
        float deltaAlpha;
        S2D = S2D / ddv;
        dbeta = 1.0 / dbeta;
        dz = dz * 0.5;
        for (int angIdx = 0; angIdx < PN; ++angIdx)
        {
            cossin = cossinZT[angIdx];
            cursour = make_float3(
                s.x * cossin.x - s.y * cossin.y,
                s.x * cossin.y + s.y * cossin.x,
                s.z + cossin.z);

            dir = curvox - cursour;
            l_square = dir.x * dir.x + dir.y * dir.y;
            l = rsqrt(l_square);
            idxU = (dir.z + dz) * S2D * l + detCntIdx.y + 1; //0.5 offset Because we use the texture fetching
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

            summ += (
                -tex3D<float>(texObj, idxD, idxR, angIdx + 0.5)
                - tex3D<float>(texObj, idxU, idxL, angIdx + 0.5)
                + tex3D<float>(texObj, idxD, idxL, angIdx + 0.5)
                + tex3D<float>(texObj, idxU, idxR, angIdx + 0.5)) * cosVal;
        }
        __syncthreads();
        vol[__umul24((__umul24(id.y, VN.x) + id.x), VN.z) + id.z] = summ;// * MSK[threadIdx.y][threadIdx.x];
    }
}


// template specialization of the pseudo DD backprojection method
template<>
__global__ void DD3_gpu_back_ker<_PSEUDODD>(
    cudaTextureObject_t texObj,
    float* vol,
    byte* __restrict__ msk,
    float3* __restrict__ cossinZT,
    float3 s,
    float S2D,
    float3 objCntIdx,
    float dx, float dz, float dbeta, float ddv,
    float2 detCntIdx, int3 VN, int PN, int squared)
{
    int k = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    int i = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    int j = __mul24(blockDim.z, blockIdx.z) + threadIdx.z;

    if (i < VN.x && j < VN.y && k < VN.z)
    {
        if (msk[j * VN.x + i] != 1)
            return;
        // Current voxel
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
            idxZ = dir.z * S2D * invl + detCntIdx.y + 0.5; //because of texture, 0.5 offset is acquired
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
        vol[(j * VN.x + i) * VN.z + k] = summ;// * MSK[threadIdx.y][threadIdx.x];
    }
}

// template specialization of the Z line based branchless DD
// NOTE: it has bugs. It requires the # of image planes is divisible by 16.
template<>
__global__ void DD3_gpu_back_ker<_ZLINEBRANCHLESS>(
    cudaTextureObject_t texObj,
    float* vol,
    byte* __restrict__ msk,
    float3* __restrict__ cossinZT,
    float3 s,
    float S2D,
    float3 objCntIdx,
    float dx, float dz, float dbeta, float ddv,
    float2 detCntIdx, int3 VN, int PN, int squared)
{
    int idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int idy = threadIdx.y + __umul24(blockIdx.y, blockDim.y);
    __shared__ float summ[4][8][16 + 1]; //for 16 layers.
#pragma unroll
    for (int i = 0; i <= 16; ++i)
    {
        summ[threadIdx.y][threadIdx.x][i] = 0;
    }
    __syncthreads();
    if (idx < VN.x && idy < VN.y)
    {
        if (msk[idy * VN.x + idx] != 1)
            return;
        float curang(0);
        float2 dirlft, dirrgh;
        //float2 dirsour;
        float3 cursour;
        float idxL, idxR, idxD;
        float cosVal = 1.0;

        float2 curvox_xy = make_float2((idx - objCntIdx.x) * dx, (idy - objCntIdx.y) * dx);
        float2 dirxy;

        int LPs = VN.z >> 4; //It has bugs because if ZN is not divisible by 16 or other values.
        float dirZ;
        //float inv_cosGamma = 0;
        float minObj = 0;
        float s2vlength = 0;
        float3 cossinT;
        S2D = S2D / ddv;
        dbeta = 1.0 / dbeta;
        float invSID = rsqrtf(s.x * s.x + s.y * s.y);
        for (int lpIdx = 0; lpIdx != LPs; ++lpIdx)  // Which subblock are now in
        {
            minObj = (-objCntIdx.z + lpIdx * 16) * dz; //Not boundary
            for (int angIdx = 0; angIdx < PN; ++angIdx)
            {
                cossinT = cossinZT[angIdx];

                cursour = make_float3(
                    s.x * cossinT.x - s.y * cossinT.y,
                    s.x * cossinT.y + s.y * cossinT.x,
                    s.z + cossinT.z);
                dirxy.x = curvox_xy.x - cursour.x;
                dirxy.y = curvox_xy.y - cursour.y;
                s2vlength = hypotf(dirxy.x, dirxy.y);
                if (fabsf(cossinT.x) <= fabsf(cossinT.y))
                {
                    dirlft = normalize(make_float2(dirxy.x, dirxy.y - 0.5 * dx));
                    dirrgh = normalize(make_float2(dirxy.x, dirxy.y + 0.5 * dx));
                    cosVal = (dx * s2vlength / dirxy.x); // Cos gamma no weighting
                }
                else
                {
                    dirlft = normalize(make_float2(dirxy.x + 0.5f * dx, dirxy.y));
                    dirrgh = normalize(make_float2(dirxy.x - 0.5f * dx, dirxy.y));
                    cosVal = (dx * s2vlength / dirxy.y);
                }

                idxL = asinf((cursour.y * dirlft.x - cursour.x * dirlft.y) * invSID) * dbeta + detCntIdx.x + 1;
                idxR = asinf((cursour.y * dirrgh.x - cursour.x * dirrgh.y) * invSID) * dbeta + detCntIdx.x + 1;

                // current pixel weighting on Z
                curang = S2D / s2vlength;
#pragma unroll
                for (int idz = 0; idz <= 16; ++idz)
                {
                    dirZ = minObj + idz * dz - cursour.z;
                    ddv = hypotf(dirZ, s2vlength) / s2vlength;
                    idxD = (dirZ - 0.5 * dz) * curang + detCntIdx.y + 1;
                    summ[threadIdx.y][threadIdx.x][idz] +=
                        (tex3D<float>(texObj, idxD, idxR, angIdx + 0.5) -
                            tex3D<float>(texObj, idxD, idxL, angIdx + 0.5)) * cosVal * ddv;
                }
            }
            __syncthreads();

            int vIdx = (idy * VN.x + idx) * VN.z + (lpIdx << 4);
#pragma unroll
            for (int idz = 0; idz < 16; ++idz)
            {
                vol[vIdx + idz] = summ[threadIdx.y][threadIdx.x][idz + 1] - summ[threadIdx.y][threadIdx.x][idz];
                summ[threadIdx.y][threadIdx.x][idz] = 0;
            }
            summ[threadIdx.y][threadIdx.x][16] = 0;
            __syncthreads();
        }
    }
}



// \brief The basic version especially that we assume the detCntIdxU and detCntIdxV
template<BackProjectionMethod Method>
void DD3_gpu_back(
    float x0, float y0, float z0,
    int DNU, int DNV,
    float* xds, float* yds, float* zds,
    float imgXCenter, float imgYCenter, float imgZCenter,
    float* hangs, float* hzPos, int PN,
    int XN, int YN, int ZN,
    float* hvol, float* hprj,
    float dx, float dz,
    byte* mask, int squared, int gpunum)
{
    CUDA_CHECK_RETURN(cudaSetDevice(gpunum));
    CUDA_CHECK_RETURN(cudaDeviceReset());

    float3 objCntIdx = make_float3(
        (XN - 1.0) * 0.5 - imgXCenter / dx,
        (YN - 1.0) * 0.5 - imgYCenter / dx,
        (ZN - 1.0) * 0.5 - imgZCenter / dz);
    float3 sour = make_float3(x0, y0, z0);
    thrust::device_vector<byte> msk(mask, mask + XN * YN);
    thrust::device_vector<float> vol(XN * YN * ZN, 0);
    const float S2D = hypotf(xds[0] - x0, yds[0] - y0);

    thrust::device_vector<float3> cossinZT(PN);
    thrust::device_vector<float> angs(hangs, hangs + PN);
    thrust::device_vector<float> zPos(hzPos, hzPos + PN);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(angs.begin(), zPos.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(angs.end(), zPos.end())),
        cossinZT.begin(),
        CTMBIR::ConstantForBackProjection<float>(x0, y0, z0)); // Calculate the constant parameters (cosine, sine, zshifts) and bind them together in cossinZT

                                                               //Calculate detCtrIdxU, detCtrIdxV, dbeta, ddv and bind them in a float4 datatype
    float4 detParas = calDetParas(xds, yds, zds, x0, y0, z0, DNU, DNV);

    cudaArray *d_prjArray = nullptr;
    cudaTextureObject_t texObj;
    dim3 blk;
    dim3 gid;
    thrust::device_vector<float> prjSAT;

    //prepare different configurations and SAT/projection data for different backprojection modes.
    switch (Method)
    {
    case _PSEUDODD:
        // case _VOLUMERENDERING: // disable school code FUL 2015-11-19
        createTextureObject<float>(texObj, d_prjArray, DNV, DNU, PN, hprj, cudaMemcpyHostToDevice,
            cudaAddressModeBorder, cudaFilterModeLinear, cudaReadModeElementType,
            false);
        blk.x = BACK_BLKX;
        blk.y = BACK_BLKY;
        blk.z = BACK_BLKZ;
        gid.x = (ZN + blk.x - 1) / blk.x;
        gid.y = (XN + blk.y - 1) / blk.y;
        gid.z = (YN + blk.z - 1) / blk.z;
        break;
    case _BRANCHLESS:
        prjSAT = genSAT_of_Projection(hprj, DNU, DNV, PN);
        createTextureObject<float>(texObj, d_prjArray, DNV + 1, DNU + 1, PN,
            thrust::raw_pointer_cast(&prjSAT[0]),
            cudaMemcpyDeviceToDevice,
            cudaAddressModeClamp, cudaFilterModeLinear, cudaReadModeElementType,
            false);
        prjSAT.clear();

        blk.x = BACK_BLKX;
        blk.y = BACK_BLKY;
        blk.z = BACK_BLKZ;
        gid.x = (ZN + blk.x - 1) / blk.x;
        gid.y = (XN + blk.y - 1) / blk.y;
        gid.z = (YN + blk.z - 1) / blk.z;
        break;
    case _ZLINEBRANCHLESS:
        prjSAT = genSAT_of_Projection(hprj, DNU, DNV, PN);
        createTextureObject<float>(texObj, d_prjArray, DNV + 1, DNU + 1, PN,
            thrust::raw_pointer_cast(&prjSAT[0]),
            cudaMemcpyDeviceToDevice,
            cudaAddressModeClamp, cudaFilterModeLinear, cudaReadModeElementType,
            false);
        prjSAT.clear();
        blk.x = 8;
        blk.y = 4;
        blk.z = 1;
        gid.x = (XN + blk.x - 1) / blk.x;
        gid.y = (YN + blk.y - 1) / blk.y;
        break;
    }
    //Back projection kernel
    DD3_gpu_back_ker<Method> << <gid, blk >> >(texObj,
        thrust::raw_pointer_cast(&vol[0]),
        thrust::raw_pointer_cast(&msk[0]),
        thrust::raw_pointer_cast(&cossinZT[0]),
        sour,
        S2D,
        objCntIdx,
        dx, dz, detParas.z, detParas.w,
        make_float2(detParas.x, detParas.y),
        make_int3(XN, YN, ZN), PN, static_cast<int>(squared != 0));

    thrust::copy(vol.begin(), vol.end(), hvol);
    destroyTextureObject(texObj, d_prjArray);

    vol.clear();
    msk.clear();
    angs.clear();
    zPos.clear();
    cossinZT.clear();
}



// assumes all detectors are of equal size in both xy and z directions (uniformly distriubed between the first and last detectors)

void DD3Back_gpu(
    float x0, float y0, float z0, // src position in gantry coordinates
    int DNU, int DNV,  // detector channel#, detector row#
    float* xds, float* yds, float* zds, // detector center position in x, y, z dimension
    float imgXCenter, float imgYCenter, float imgZCenter, // img center in world coordinate
    float* hangs, float* hzPos, int PN, // view angles, src position in Z in world coordinate, view#
    int XN, int YN, int ZN, // dimension of image volume
    float* hvol, float* hprj, // img data, projection data
    float dx, float dz, // img size in xy, img size in z
    byte* mask, int gpunum, int squared, int prjMode)
{

    switch (prjMode)
    {
    case 0:
        DD3_gpu_back<_BRANCHLESS>(x0, y0, z0, DNU, DNV,
            xds, yds, zds,
            imgXCenter, imgYCenter, imgZCenter,
            hangs, hzPos, PN, XN, YN, ZN,
            hvol, hprj, dx, dz, mask, squared, gpunum);
        break;
    case 1:
        DD3_gpu_back<_PSEUDODD>(x0, y0, z0, DNU, DNV,
            xds, yds, zds,
            imgXCenter, imgYCenter, imgZCenter,
            hangs, hzPos, PN, XN, YN, ZN,
            hvol, hprj, dx, dz, mask, squared, gpunum);
        break;
    case 2:
        DD3_gpu_back<_PSEUDODD>(x0, y0, z0, DNU, DNV,
            xds, yds, zds,
            imgXCenter, imgYCenter, imgZCenter,
            hangs, hzPos, PN, XN, YN, ZN,
            hvol, hprj, dx, dz, mask, squared, gpunum);
        break;
    case 3:
        DD3_gpu_back<_ZLINEBRANCHLESS>(x0, y0, z0, DNU, DNV,
            xds, yds, zds,
            imgXCenter, imgYCenter, imgZCenter,
            hangs, hzPos, PN, XN, YN, ZN,
            hvol, hprj, dx, dz, mask, squared, gpunum);
        break;
    default:
        DD3_gpu_back<_BRANCHLESS>(x0, y0, z0, DNU, DNV,
            xds, yds, zds,
            imgXCenter, imgYCenter, imgZCenter,
            hangs, hzPos, PN, XN, YN, ZN,
            hvol, hprj, dx, dz, mask, squared, gpunum);
        break;
    }
}




////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace CTMBIR
{
    void DD3_GPU_help_mex()
    {
        std::cout << "Usage DD3_GPU_Proj:\n"
            << "y = function('Proj', x0, y0, z0, nrdetcols, nrdetrows, *xds, *yds, *zds,*xdsl,*ydsl,*xdsr,*ydsr, imgXoffset, imgYoffset, imgZoffset, *viewangles, *zshifts, nrviews, nrcols, nrrows, nrplanes, *pOrig, vox_xy_size, vox_zsize, xy_mask, gpunum)\n"
            << "float x0, source x coordinate (before rotating)\n"
            << "float y0, source y coordinate (before rotating)\n"
            << "float z0, source z coordinate (before rotating)\n"
            << "int nrdetcols, number of detector columns (in-plane)\n"
            << "int nrdetrows, number of detector rows (in-z)\n"
            << "float* xds, nrdetcols detector x coordinates (before rotating)\n"
            << "float* yds, nrdetcols detector y coordinates (before rotating)\n"
            << "float* zds, nrdetrows detector z coordinates (before rotating)\n"
            << "float imgXoffsets, the x-offset of the image center relative to COR\n"
            << "float imgYoffsets, the y-offset of the image center relative to COR\n"
            << "float imgZoffsets, the z-offset of the image center relative to COR\n"
            << "float* viewangles, nrviews rotation angles\n"
            << "float* zshifts, nrviews z-increments of the patient/grantry(for helical)\n"
            << "int nrviews, number of angles\n"
            << "int nrcols, number of columns in image\n"
            << "int nrrows, number of rows in image\n"
            << "int nrplanes, number of planes in image\n"
            << "float* pOrig, least significant index is plane, then row, then col\n" // What is that?
            << "float vox_xy_size, voxel size in x and y\n"
            << "float vox_z_size, voxel size in z\n"
            << "byte* xy_mask, xy plane [mask, mask_trans]\n"
            << "int gpunum, multi GPU in views\n"
            << "int prjMode, Different projection modes"
            << "out: [det-col, det-row, view] sinogram\n"
            << "no need to input zero sinogram\n"
            << "y = function('Back', x0, y0, z0, nrdetcols, nrdetrows, *xds, *yds, *zds, imgXoffset, imgYoffset, imgZoffset, *viewangles,*zshifts,nrviews,*sinogram,nrcols,nrrows,nrplanes,vox_xy_size,vox_z_size,xy_mask,gpunum,projector_type)\n"
            << "int projector_type: 0 corresponds to a standard backprojector\n"
            << "                    1 corresponds to a squared backprojector\n"
            << "float* sinogram, least significant index is det-row, then det-col then view\n"
            << "out: [plane row col] image\n"
            << "no need to input zero image\n";
    }
}


void DD3_GPU_Proj_mex(
    mxArray* plhs[], // [view det-row det-col]
    const mxArray* mx_x0, //source x coordinate (before rotating)
    const mxArray* mx_y0, //source y coordinate (before rotating)
    const mxArray* mx_z0, //source z coordinate (before rotating)
    const mxArray* mx_nrdetcols, //number of detector columns (in-plane)
    const mxArray* mx_nrdetrows, //number of detector rows (in-Z)
    const mxArray* mx_xds, //nrdetcols detector x coordinates (before rotating)
    const mxArray* mx_yds, //nrdetcols detector y coordinates (before rotating)
    const mxArray* mx_zds, //nrdetrows detector z coordinates (before rotating)
    const mxArray* mx_imgXoffset, //the x-offset of the image center relative to COR
    const mxArray* mx_imgYoffset, //the y-offset of the image center relative to COR
    const mxArray* mx_imgZoffset, //the z-offset of the image center relative to COR
    const mxArray* mx_viewangles, //nrviews rotation angles
    const mxArray* mx_zshifts, //nrviews z-increments of the patient gantry (for helical)
    const mxArray* mx_nrviews, //number of angles
    const mxArray* mx_nrcols, // number of columns in image
    const mxArray* mx_nrrows, // number of rows in image
    const mxArray* mx_nrplanes, //number of planes in image
    const mxArray* mx_pOrig, //least significant index is plane, then row, then col
    const mxArray* mx_vox_xy_size, //voxel size in x and y
    const mxArray* mx_vox_z_size, //voxel size in Z
    const mxArray* mx_xy_mask, //xy plane [mask, mask_trans]
    const mxArray* mx_gpunum,//multi gpu in views
    const mxArray* mx_prjMode)  //projection modes
{
    int nrdetcols = *((int*)mxGetData(mx_nrdetcols));
    int nrdetrows = *((int*)mxGetData(mx_nrdetrows));
    int nrviews = *((int*)mxGetData(mx_nrviews));
    //Create output array of class mxREAL and mxSINGLE_CLASS
    const mwSize dims[] = { nrdetrows, nrdetcols, nrviews };
    plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

    float* xds = (float*)mxGetPr(mx_xds);
    float* yds = (float*)mxGetPr(mx_yds);
    float* zds = (float*)mxGetPr(mx_zds);

    DD3Proj_gpu(
        *((float*)mxGetData(mx_x0)),
        *((float*)mxGetData(mx_y0)),
        *((float*)mxGetData(mx_z0)),
        nrdetcols, nrdetrows,
        xds,
        yds,
        zds,
        *((float*)mxGetData(mx_imgXoffset)),
        *((float*)mxGetData(mx_imgYoffset)),
        *((float*)mxGetData(mx_imgZoffset)),
        (float*)mxGetPr(mx_viewangles),
        (float*)mxGetPr(mx_zshifts),
        *((int*)mxGetData(mx_nrviews)),
        *((int*)mxGetData(mx_nrcols)),
        *((int*)mxGetData(mx_nrrows)),
        *((int*)mxGetData(mx_nrplanes)),
        (float*)mxGetPr(mx_pOrig),
        (float*)mxGetPr(plhs[0]),
        *((float*)mxGetData(mx_vox_xy_size)),
        *((float*)mxGetData(mx_vox_z_size)),
        (byte*)mxGetPr(mx_xy_mask),
        *((int*)mxGetData(mx_gpunum)),
        *((int*)mxGetData(mx_prjMode)));


}



/**
* DD3_GPU_Back_mex()
*/
void DD3_GPU_Back_mex(
    mxArray* plhs[], //[plane, XN(row), YN(col)]
    const mxArray* mx_x0, //source x coordinate (before rotating)
    const mxArray* mx_y0, //source y coordinate (before rotating)
    const mxArray* mx_z0, //source z coordinate (before translating)
    const mxArray* mx_nrdetcols, //number of detector columns (in-plane)
    const mxArray* mx_nrdetrows, //number of detector rows(in-z)
    const mxArray* mx_xds, //nrdetcols detector x coordinate (before rotating)
    const mxArray* mx_yds, //nrdetcols detector y coordinate (before rotating)
    const mxArray* mx_zds, //nrdetrows detector z coordinate (before translating)
    const mxArray* mx_imgXoffset, // the x-offset of the image center relative to COR
    const mxArray* mx_imgYoffset, // the y-offset of the image center relative to COR
    const mxArray* mx_imgZoffset, // the z-offset of the image center relative to COR
    const mxArray* mx_viewangles, //nrviews rotation angles
    const mxArray* mx_zshifts, //nrviews z-increments ofthe patient/gantry (for helical)
    const mxArray* mx_nrviews, //number of angles,
    const mxArray* mx_nrcols, //# columns in image
    const mxArray* mx_nrrows, //# rows in image
    const mxArray* mx_nrplanes, //# planes in image
    const mxArray* mx_sinogram, //# least significant index is view, then det-row, then det-col
    const mxArray* mx_vox_xy_size, //voxel size in x and y
    const mxArray* mx_vox_z_size, //voxel size in z
    const mxArray* mx_xy_mask, // xy plane [mask/ no trans!]
    const mxArray* mx_gpunum, // how many gpu would you like to use,the maximum number is 2 and the minimum is 1
    const mxArray* mx_projector_type, //0 : standard backprojector; 1: squared backprojector
    const mxArray* mx_prjMode) // we would like to use string/char*
{
    int nrcols = *((int*)mxGetData(mx_nrcols));
    int nrrows = *((int*)mxGetData(mx_nrrows));
    int nrplanes = *((int*)mxGetData(mx_nrplanes));

    int nrdetcols = *((int*)mxGetData(mx_nrdetcols));
    int nrdetrows = *((int*)mxGetData(mx_nrdetrows));
    int nrviews = *((int*)mxGetData(mx_nrviews));

    const mwSize dims[] = { nrplanes, nrrows, nrcols };
    plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

    float* xds = (float*)mxGetPr(mx_xds);
    float* yds = (float*)mxGetPr(mx_yds);
    float* zds = (float*)mxGetPr(mx_zds);

    DD3Back_gpu(
        *((float*)mxGetData(mx_x0)),
        *((float*)mxGetData(mx_y0)),
        *((float*)mxGetData(mx_z0)),
        nrdetcols, nrdetrows,
        xds, yds, zds,
        *((float*)mxGetData(mx_imgXoffset)),
        *((float*)mxGetData(mx_imgYoffset)),
        *((float*)mxGetData(mx_imgZoffset)),
        (float*)mxGetPr(mx_viewangles),
        (float*)mxGetPr(mx_zshifts),
        *((int*)mxGetData(mx_nrviews)),
        *((int*)mxGetData(mx_nrcols)),
        *((int*)mxGetData(mx_nrrows)),
        *((int*)mxGetData(mx_nrplanes)),
        (float*)(mxGetPr(plhs[0])),
        (float*)mxGetPr(mx_sinogram),
        *((float*)mxGetData(mx_vox_xy_size)),
        *((float*)mxGetData(mx_vox_z_size)),
        (byte*)mxGetPr(mx_xy_mask),
        *((int*)mxGetData(mx_gpunum)),
        *((int*)mxGetData(mx_projector_type)),
        *((int*)mxGetData(mx_prjMode)));
}



/*
* DD3_GPU_mex()
*/
void DD3_GPU_mex(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 1)
    {
        CTMBIR::DD3_GPU_help_mex();
        std::cerr << "Error: usage\n";
    }

    if (!mxIsChar(prhs[0]))
    {
        std::cerr << "First argument must be a string\n";
        CTMBIR::DD3_GPU_help_mex();
        std::cerr << "Error: usage\n";
    }

    //Read and copy first in-out string to arg[0]
    int arg0len = mxGetM(prhs[0]) * mxGetN(prhs[0]) + 1;
    char* arg0 = (char*)mxCalloc(arg0len, sizeof(char));
    if (mxGetString(prhs[0], arg0, arg0len))
    {
        std::cerr << "but with mxGetString\n";
    }

    //Forward projection
    if (!std::strcmp(arg0, "Proj"))
    {
        if (nrhs != 24 || nlhs != 1)
        {
            CTMBIR::DD3_GPU_help_mex();
            std::cerr << "Error: usage\n";
        }

        DD3_GPU_Proj_mex(plhs, prhs[1], prhs[2], prhs[3], prhs[4], prhs[5], prhs[6], prhs[7], prhs[8], prhs[9], prhs[10], prhs[11], prhs[12], prhs[13], prhs[14], prhs[15], prhs[16], prhs[17], prhs[18], prhs[19], prhs[20], prhs[21], prhs[22], prhs[23]);
    }
    else if (!std::strcmp(arg0, "Back"))
    {
        if (nrhs != 25 || nlhs != 1)
        {
            CTMBIR::DD3_GPU_help_mex();
            std::cerr << "Error: usage\n";
        }

        DD3_GPU_Back_mex(plhs, prhs[1], prhs[2], prhs[3], prhs[4], prhs[5], prhs[6], prhs[7], prhs[8], prhs[9], prhs[10], prhs[11], prhs[12], prhs[13], prhs[14], prhs[15], prhs[16], prhs[17], prhs[18], prhs[19], prhs[20], prhs[21], prhs[22], prhs[23], prhs[24]);
    }
    else
    {
        std::cerr << "Error: unsupported action " << (const char*)mxGetData(prhs[0]) << "\n";
    }

}


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (!nlhs && !nrhs)
    {
        CTMBIR::DD3_GPU_help_mex();
        return;
    }
    DD3_GPU_mex(nlhs, plhs, nrhs, prhs);
}
