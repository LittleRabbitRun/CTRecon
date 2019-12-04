/*
**  CT/Micro CT lab
**  Department of Radiology
**  University of Iowa
**  Version of 2004.03.05
*/

#include <mex.h>
#include <math.h>

typedef struct {
    float *image;
    const float *proj;
    double *work_space;
} FloatReconData;

struct DoubleReconData{
    double *image;
    const double *proj;
    double *work_space;
};

static const double *geom[3];

#define PI 3.14159265358979
#define DeltaMax 0.0000001


extern "C"
void backProjection(
	double* hImage, const double* hProj,
	int RecMX, int RecMY, int RecMZ,
	double ObjRSquare,
	double* hxCor, double* hyCor, double* hzCor,
	double* hVectorS, // projectionNum * 3
	double* hVectorE1, // projectionNum * 2
	double* hVectorE2, // projectionNum * 2
	double ScanR,
	double StdDis,
	double DeltaU, double DeltaV,
	double HalfY, double HalfZ,
	int YL, int YLZL,
	double HelicP, double DeltaL, double ProjCtr,
	double ProjBeginIndex,
	int ProjNum, // number of projections
	double ProjScale,
	int threadidx, int threadidy, int threadidz);

inline double bilinearInterpolation(double YY, double ZZ, int YLZL, int YL, int ProjIndex, const double* Proj)
{
    int YYD = int(YY);
    int YYU = YYD + 1;
    int ZZD = int(ZZ);
    int ZZU = ZZD + 1;
    
    double alfa = YY - YYD;
    double beta = ZZ - ZZD;
    return (Proj[ProjIndex*YLZL+ZZD*YL+YYD]*(1-alfa)*(1-beta)+
          Proj[ProjIndex*YLZL+ZZD*YL+YYU]*alfa*(1-beta)+
          Proj[ProjIndex*YLZL+ZZU*YL+YYD]*(1-alfa)*beta+
          Proj[ProjIndex*YLZL+ZZU*YL+YYU]*alfa*beta);
}
struct double3
{
    double x;
    double y;
    double z;
};

void PISegment(double x,double y,double z, double &BAngle, double &TAngle);
void VoxelDrivenBkProj(DoubleReconData *data)
{
    /*Compute system parameter*/
    double ScanR,ObjR,StdDis,HelicP,DecWidth, DecHeigh,ProjCtr;
    int  ProjScale, ProjBeginIndex,ProjEndIndex,ProjNum, YL, ZL, RecMX, RecMY, RecMZ;
    
    //ScanR  StdDis HelicP ProjScale ProjBeginIndex ProjEndIndex ProjCenter
    //DecWidth YL DecHeigh ZL
    //ObjR RecSize RecSize RecSize
    ScanR    = geom[0][0];
    ObjR     = geom[2][0];
    StdDis   = geom[0][1];
    HelicP   = geom[0][2];
    DecWidth = geom[1][0];
    DecHeigh = geom[1][2];
    ProjCtr  = geom[0][6];
    ProjScale= int(geom[0][3]);
    ProjBeginIndex = int(geom[0][4]);
    ProjEndIndex   = int(geom[0][5]);
    ProjNum  = ProjEndIndex-ProjBeginIndex+1;
    YL       = int(geom[1][1]);
    ZL       = int(geom[1][3]);
    RecMX    = int(geom[2][1]);
    RecMY    = int(geom[2][2]);
    RecMZ    = int(geom[2][3]);
    
    mexPrintf("ProjBeginIndex=%d, ProjEndIndex=%d \n\t",ProjBeginIndex,ProjEndIndex);
    
    
    /*Compute some often used constant*/
    double DeltaL, DeltaU, DeltaV, DeltaX, DeltaY, DeltaZ;
    double HalfY, HalfZ, XCenter, YCenter, ZCenter;
    
    DeltaL = 2*PI/ProjScale;
    DeltaU = DecWidth/YL;
    DeltaV = DecHeigh/ZL;
    HalfY  = (YL-1)*0.5;
    HalfZ  = (ZL-1)*0.5;
    DeltaX = 2*ObjR/RecMX;
    DeltaY = 2*ObjR/RecMY;
    DeltaZ = 2*ObjR/RecMZ;
    XCenter= (RecMX-1)*0.5;
    YCenter= (RecMY-1)*0.5;
    ZCenter= (RecMZ-1)*0.5;
    
    /*Compute the coordinates of scanning locus and its corresponding local coordinate*/
    
    double  *VectorS, *VectorE1,*VectorE2,*xCor,*yCor,*zCor,*Coef;
    int  loop;
    double temp;
    
    VectorS  = (double*)mxCalloc(3*ProjNum,sizeof(double));
    VectorE1 = (double*)mxCalloc(2*ProjNum,sizeof(double));
    VectorE2 = (double*)mxCalloc(2*ProjNum,sizeof(double));
    xCor     = (double*)mxCalloc(RecMX,sizeof(double));
    yCor     = (double*)mxCalloc(RecMY,sizeof(double));
    zCor     = (double*)mxCalloc(RecMZ,sizeof(double));
    Coef     = (double*)mxCalloc(ProjNum,sizeof(double));
    
    for(loop=0;loop<ProjNum;loop++)
    {
        temp = (loop+ProjBeginIndex-ProjCtr)*DeltaL;
        VectorS[loop*3  ] = ScanR*cos(temp);
        VectorS[loop*3+1] = ScanR*sin(temp);
        VectorS[loop*3+2] = HelicP*temp/(2*PI);
        VectorE1[loop*2  ]=-cos(temp);
        VectorE1[loop*2+1]=-sin(temp);
        VectorE2[loop*2  ]=-sin(temp);
        VectorE2[loop*2+1]= cos(temp);
    }
    
    for(loop=0;loop<RecMX;loop++)
        xCor[loop] = (loop-XCenter)*DeltaX;
    for(loop=0;loop<RecMY;loop++)
        yCor[loop] = (loop-YCenter)*DeltaY;
    for(loop=0;loop<RecMZ;loop++)
        zCor[loop] = (loop-ZCenter)*DeltaZ;
    
    //BackProjection to reconstruct the object
    int  Xindex,Yindex,Zindex,Bindex,Tindex,ProjIndex,LoopBegin,LoopEnd;
    int  YYU, YYD, ZZU, ZZD, YLZL, RecMXY;
    double ObjRSquare, x, y, z, BAngle, TAngle, tpdata;
    double DPSx, DPSy, DPSz, factor, fenmu, YY, ZZ, alfa, beta;
    const double *Proj;
    double  *Image;
    
    YLZL   = YL*ZL;
    RecMXY = RecMX*RecMY;
    Image  = data->image;
    Proj   = data->proj;
    ObjRSquare = pow(ObjR,2);
    
    
     for (Xindex=0; Xindex<RecMX; Xindex++)
        {
            for(Yindex=0; Yindex<RecMY; Yindex++)
            {
                if( pow(xCor[Xindex],2) + pow(yCor[Yindex],2)< ObjRSquare)
                {
                    for (Zindex=0; Zindex<RecMZ; Zindex++)
                        //+ pow(zCor[Zindex],2) < ObjRSquare)
                    {
                        x = xCor[Xindex]/ScanR;
                        y = yCor[Yindex]/ScanR;
                        z = zCor[Zindex]/HelicP;
                        PISegment(x,y,z,BAngle,TAngle);
                        BAngle = BAngle/DeltaL+ProjCtr-ProjBeginIndex;
                        TAngle = TAngle/DeltaL+ProjCtr-ProjBeginIndex;
                        
                        if(!((BAngle>=ProjNum-1)||(TAngle<=0)))
                        {
                            Bindex = int(BAngle+10)-10;
                            Tindex = int(TAngle+10)-9;
                            for (loop=0; loop< ProjNum; loop++)
                                Coef[loop] = 1;
                            
                            if(Bindex<0)
                            {
                                LoopBegin = 1;
                            }
                            else if(Bindex==0)
                            {
                                temp    = 1-BAngle+Bindex;
                                Coef[1] = 0.5+(1-temp*0.5)*temp;
                                LoopBegin = 1;
                            }
                            else if(Bindex==ProjNum-2)
                            {
                                temp    = 1-BAngle+Bindex;
                                Coef[ProjNum-2] = pow(temp,2)*0.5;
                                LoopBegin = ProjNum-2;
                            }
                            else
                            {
                                temp    = 1-BAngle+Bindex;
                                Coef[Bindex] = pow(temp,2)*0.5;
                                Coef[Bindex+1] = 0.5+(1-temp*0.5)*temp;
                                LoopBegin = Bindex;
                            }
                            ///////////////////////////////////
                            if(Tindex>ProjNum-1)
                            {
                                LoopEnd = ProjNum-2;
                                
                            }
                            else if(Tindex==ProjNum-1)
                            {
                                temp    = 1+TAngle-Tindex;
                                Coef[ProjNum-2]= 0.5+(1-temp*0.5)*temp;
                                LoopEnd = ProjNum-2;
                            }
                            else if(Tindex==1)
                            {
                                LoopEnd = 1;
                                temp    = 1+TAngle-Tindex;
                                Coef[1]  = pow(temp,2)*0.5;
                            }
                            else
                            {
                                LoopEnd = Tindex;
                                temp    = 1+TAngle-Tindex;
                                Coef[Tindex-1]= 0.5+(1-temp*0.5)*temp;
                                Coef[Tindex]  = pow(temp,2)*0.5;
                            }
                            
                            tpdata = 0;
                            for(ProjIndex=LoopBegin;ProjIndex<=LoopEnd;ProjIndex++)
                            {
                                DPSx  = xCor[Xindex]-VectorS[ProjIndex*3];
                                DPSy  = yCor[Yindex]-VectorS[ProjIndex*3+1];
                                DPSz  = zCor[Zindex]-VectorS[ProjIndex*3+2];
                                factor= sqrt(DPSx*DPSx+DPSy*DPSy+DPSz*DPSz);
                                fenmu = DPSx*VectorE1[ProjIndex*2]+DPSy*VectorE1[ProjIndex*2+1];
                                YY  = DPSx*VectorE2[ProjIndex*2]+DPSy*VectorE2[ProjIndex*2+1];
                                YY  = YY*StdDis/(fenmu*DeltaU)+HalfY;
                                ZZ  = DPSz*StdDis/(fenmu*DeltaV)+HalfZ;
                                YYD = int(YY);
                                YYU = YYD+1;
                                ZZD = int(ZZ);
                                ZZU = ZZD+1;
                                alfa= YY-YYD;
                                beta= ZZ-ZZD;
                                temp= Proj[ProjIndex*YLZL+ZZD*YL+YYD]*(1-alfa)*(1-beta)+
                                        Proj[ProjIndex*YLZL+ZZD*YL+YYU]*alfa*(1-beta)+
                                        Proj[ProjIndex*YLZL+ZZU*YL+YYD]*(1-alfa)*beta+
                                        Proj[ProjIndex*YLZL+ZZU*YL+YYU]*alfa*beta;
                                //YYD = int(YY+0.5);
                                //ZZD = int(ZZ+0.5);
                                //temp = Proj[ProjIndex*YLZL+ZZD*YL+YYD];
                                temp= temp*Coef[ProjIndex];
                                tpdata= tpdata+temp/factor;
                            }
                            tpdata=-tpdata/ProjScale;
                            ProjIndex=(Xindex * RecMY + Yindex) * RecMZ + Zindex;
                            Image[ProjIndex] = tpdata;
                        }//if((BAngle>0)&(TAngle<ProjNum-1))
                    }//if( pow(xCor[Xindex],2) + pow(yCor[Yindex],2)...
                }
            }
        }

    
    //Free the the mxArray variable
    mxFree(VectorS);
    mxFree(VectorE1);
    mxFree(VectorE2);
    mxFree(xCor);
    mxFree(yCor);
    mxFree(zCor);
    mxFree(Coef);
}

void PISegment(double x,double y,double z, double &BAngle, double &TAngle)
{
    double delta, dm, LanbudaInital, bmin, bmax, r2;
    double sb,st,t,tempcos,templan,zz;
    
    delta=1;
    dm = DeltaMax;
    LanbudaInital=z*2*PI;
    bmin = LanbudaInital-2*PI;
    bmax = LanbudaInital;
    r2   = x*x+y*y;
    
    while (((bmax-bmin)>dm) && (delta>dm))
    {
        sb=(bmax+bmin)*0.5;
        tempcos=2*(1-y*sin(sb)-x*cos(sb));
        t=(1-r2)/tempcos;
        templan=acos((y*cos(sb)-x*sin(sb))/sqrt(tempcos+r2-1));
        st=2*templan+sb;
        zz=(sb*t+(1-t)*st)/(2*PI);
        if(zz<z)
            bmin=sb;
        else
            bmax=sb;
        delta=fabs(zz-z);
    }
    
    BAngle = sb;
    TAngle = st;
}

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    int dims[3];

    /* Check for proper number of arguments */
    if (nrhs != 3) {
        mexErrMsgTxt("bkproj requires two input arguments.");
    }
    if (nlhs != 0) {
        mexErrMsgTxt("bkproj requires one output argument.");
    }


    /* argument about scanning locus: ScanR, StdDis, HelicP, ProjScale, ProjNumber */
	geom[0] = mxGetPr(mxGetFieldByNumber(prhs[0], 0, 0));

    /* argument about detector: DecWidth, YL, DecHeigh, ZL */
    geom[1] = mxGetPr(mxGetFieldByNumber(prhs[0], 0, 1));

	/* argument about object:  ObjR RecMX RecMY RecMZ*/
	geom[2] = mxGetPr(mxGetFieldByNumber(prhs[0], 0, 2));
	
    if (mxGetClassID(prhs[1]) == mxSINGLE_CLASS) 
	{
        FloatReconData data;
        mexErrMsgTxt("Single precision is not supported in this version.\n");
    } 
	else 
	{
        DoubleReconData data;
        /* Assign pointers to the various parameters */
        data.image = mxGetPr(prhs[2]);
        data.proj  = mxGetPr(prhs[1]);
        /* Do the actual computations */
        VoxelDrivenBkProj(&data);
    }
}
