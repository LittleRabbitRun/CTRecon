/*
 * Wake Forest University Health Sciences reserves all copyrights
 * Organization:
 *  Wake Forest University Health Sciences
 * DD_MultiGPU_mex.cpp
 * MATLAB mex gateway routine for multi GPU based helical projection
 * and backprojection
 * Author: Rui Liu
 * Date: 2016.09.13
 * Version: 1.0
 */

#include "mex.h"
#include "matrix.h"

#include "DD_MultiGPU_ker.h"
#include <iostream>

typedef unsigned char byte;
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
	byte* mask, int prjMode, const int* startPN, int gpuNum);

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
	byte* mask, int bakMode,const int* startVOL, int gpuNum);

void DD_MultiGPU_mex(
    mxArray* plhs[],
    const mxArray* mx_x0,
    const mxArray* mx_y0,
    const mxArray* mx_z0,
    const mxArray* mx_DNU,
    const mxArray* mx_DNV,
    const mxArray* mx_xds, // ptr
    const mxArray* mx_yds, // ptr
    const mxArray* mx_zds, // ptr
    const mxArray* mx_imgXCenter,
    const mxArray* mx_imgYCenter,
    const mxArray* mx_imgZCenter,
    const mxArray* mx_hangs, // ptr
    const mxArray* mx_hzPos, // ptr
    const mxArray* mx_PN,
    const mxArray* mx_XN,
    const mxArray* mx_YN,
    const mxArray* mx_ZN,
    const mxArray* mx_input, // ptr
    const mxArray* mx_dx,
    const mxArray* mx_dz,
    const mxArray* mx_mask, // ptr
    const mxArray* mx_prjMode,
    const mxArray* mx_startPN, // ptr
    const mxArray* mx_gpuNum)
{
    const float x0 = *((float*)mxGetData(mx_x0));
    const float y0 = *((float*)mxGetData(mx_y0));
    const float z0 = *((float*)mxGetData(mx_z0));
    const int DNU = *((int*)mxGetData(mx_DNU));
    const int DNV = *((int*)mxGetData(mx_DNV));
    float* xds = (float*)mxGetPr(mx_xds);
    float* yds = (float*)mxGetPr(mx_yds);
    float* zds = (float*)mxGetPr(mx_zds);
    const float imgXCenter = *((float*)mxGetData(mx_imgXCenter));
    const float imgYCenter = *((float*)mxGetData(mx_imgYCenter));
    const float imgZCenter = *((float*)mxGetData(mx_imgZCenter));
    
    float* hangs = (float*)mxGetPr(mx_hangs);
    float* hzPos = (float*)mxGetPr(mx_hzPos);
    
    const int PN = *((int*)mxGetData(mx_PN));
    const int XN = *((int*)mxGetData(mx_XN));
    const int YN = *((int*)mxGetData(mx_YN));
    const int ZN = *((int*)mxGetData(mx_ZN));
    
    float* input = (float*)mxGetPr(mx_input);
    
    
    const float dx = *((float*)mxGetData(mx_dx));
    const float dz = *((float*)mxGetData(mx_dz));
    
    byte* mask = (byte*)mxGetPr(mx_mask);
    
    const int prjMode = *((int*)mxGetData(mx_prjMode));
    int* startPN = (int*)mxGetPr(mx_startPN);
    
    const int gpuNum = *((int*)mxGetData(mx_gpuNum));
    
    float* output = NULL;
    mwSize dims[3];
    switch(prjMode)
    {
        case 0: // Branchless DD projection
            dims[0] = DNV;
            dims[1] = DNU;
            dims[2] = PN;
            plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
            output = (float*)mxGetPr(plhs[0]);
            DD3Proj_multiGPU(x0, y0, z0, DNU, DNV, xds, yds, zds,
                imgXCenter, imgYCenter, imgZCenter, hangs, hzPos, 
                PN, XN, YN, ZN, input, output, dx, dz, mask, 0, 
                startPN, gpuNum);
            break;
        case 1: // Pseudo DD projection
            dims[0] = DNV;
            dims[1] = DNU;
            dims[2] = PN;
            plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
            output = (float*)mxGetPr(plhs[0]);
            DD3Proj_multiGPU(x0, y0, z0, DNU, DNV, xds, yds, zds,
                imgXCenter, imgYCenter, imgZCenter, hangs, hzPos, 
                PN, XN, YN, ZN, input, output, dx, dz, mask, 1, 
                startPN, gpuNum);
            break;
        case 2: // Branchless DD backprojection
            dims[0] = ZN;
            dims[1] = XN;
            dims[2] = YN;
            plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
            output = (float*)mxGetPr(plhs[0]);            
            DD3Back_multiGPU(x0, y0, z0, DNU, DNV, xds, yds, zds,
                imgXCenter, imgYCenter, imgZCenter, hangs, hzPos, 
                PN, XN, YN, ZN, output, input, dx, dz, mask, 0,
                startPN, gpuNum);
            break;
        case 3: // Pseudo DD backprojection
            dims[0] = ZN;
            dims[1] = XN;
            dims[2] = YN;
            plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
            output = (float*)mxGetPr(plhs[0]);
            DD3Back_multiGPU(x0, y0, z0, DNU, DNV, xds, yds, zds,
                imgXCenter, imgYCenter, imgZCenter, hangs, hzPos, 
                PN, XN, YN, ZN, output, input, dx, dz, mask, 1,
                startPN, gpuNum);
            break;
        default: // Do Nothing
            break;
    }
    
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs,const mxArray* prhs[])
{
    if(nlhs != 1 || nrhs != 24)
    {
        std::cerr<<"Require exactly one output\n";
        exit(-1);
    }
    DD_MultiGPU_mex(plhs,
            prhs[0],prhs[1],prhs[2],prhs[3],prhs[4],prhs[5],prhs[6],prhs[7],
            prhs[8],prhs[9],prhs[10],prhs[11],prhs[12],prhs[13],
            prhs[14],prhs[15],prhs[16],prhs[17],prhs[18],prhs[19],prhs[20], 
            prhs[21],prhs[22],prhs[23]); // gpuNum
    //mxArray* plhs[],
    //const mxArray* mx_x0,
    //const mxArray* mx_y0,
    //const mxArray* mx_z0,
    //const mxArray* mx_DNU,
    //const mxArray* mx_DNV,
    //const mxArray* mx_xds, // ptr
    //const mxArray* mx_yds, // ptr
    //const mxArray* mx_zds, // ptr
    //const mxArray* mx_imgXCenter,
    //const mxArray* mx_imgYCenter,
    //const mxArray* mx_imgZCenter,
    //const mxArray* mx_hangs, // ptr
    //const mxArray* mx_hzPos, // ptr
    //const mxArray* mx_PN, // parameter = 13
    //const mxArray* mx_XN,
    //const mxArray* mx_YN,
    //const mxArray* mx_ZN,
    //const mxArray* mx_input, // ptr paramter = 17
    //const mxArray* mx_dx,
    //const mxArray* mx_dz,
    //const mxArray* mx_mask, // ptr
    //const mxArray* mx_prjMode,
    //const mxArray* mx_startPN, // ptr
    //const mxArray* mx_gpuNum)

}
