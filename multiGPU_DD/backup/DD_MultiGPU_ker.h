#ifndef _DD_MULTIGPU_KER_H_
#define _DD_MULTIGPU_KER_H_

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

#endif