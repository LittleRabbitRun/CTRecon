function RecMatrix=FunctionKat_v6(COE,FilteringMode,Phantom, keepIntermediateResults, waitBarHandle)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  COE:  Filtering coefficent Tau
%  FilteringMode: 0=Single operator 
%                 1=Dual   operators
%                 2=Dual symmetric operators 
%  Phantom:  0=Disk phantom
%            1=Head phantom
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Interative version of Katsevich algorithm 
%   CT/Micro CT Lab
%   Department of Radiology
%   University of Iowa
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Time = cputime;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ProjNumber= 3501; % Total Projection Samping number
ScanR =75;       % (cm) Scanning radius
StdDis=150;      % (cm) Source to detector distance
HelicP= 12.5;      % (cm) Helical pitch
ObjR  =25;       % (cm) Object radius
ProjScale = 768; % Number of projection per turn
ProjCenter= (ProjNumber + 1) / 2; % 
FilterCoeFF= COE;
YL = 300;        % sampling number per row
ZL = 40;         % sampling number per column
%DecWidth =107.8;% (cm) Width of detector array
%DecHeigh =39.1; % (cm) Heigh of detector array
StepProjNumber= 3502;% Projections number per intertiative
RecSX         = 512;% Recconstruction size of the object
RecSY         = 512;
RecSZ         = 512;
DeltaX = 2.0 * ObjR / RecSX;
DeltaY = 2.0 * ObjR / RecSY;
DeltaZ = 2.0 * ObjR / RecSZ;

XCtrIdx = (RecSX - 1.0) * 0.5;
YCtrIdx = (RecSY - 1.0) * 0.5;
ZCtrIdx = (RecSZ - 1.0) * 0.5;
% 
% 	double XCenter = (RecMx - 1.0) * 0.5;
% 	double YCenter = (RecMy - 1.0) * 0.5;
% 	double ZCenter = (RecMz - 1.0) * 0.5;

DerMode       = 1; %% "1"  chain rule
                  %% "0"  direct  method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PI=3.1415926;
U_HelicP = HelicP/ObjR;
U_ScanR  = ScanR/ObjR;
U_DW     = 2*(U_ScanR/(U_ScanR^2-1)^0.5)*YL/(YL-2);
delta    =2*acos(ObjR/ScanR);
coef     =(2*PI-delta)/(2*PI*(1-cos(delta)));
U_DH     = 2*coef*U_HelicP*ZL/(ZL-2);
DecWidth = U_DW*StdDis/U_ScanR; % (cm) Width of detector array
DecHeigh = U_DH*StdDis/U_ScanR; % (cm) Heigh of detector array


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Proj     = zeros(YL,ZL,StepProjNumber);
RecMatrix= zeros(RecSX,RecSY,RecSZ);
ProjBeginIndex=1;
ProjEndIndex  = StepProjNumber;
while (ProjBeginIndex<ProjNumber)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Compute the cone projection
    ProjBeginIndex
    waitbar(0.25, waitBarHandle, 'Computing the cone beam projection');
    for ProjIndex=ProjBeginIndex:ProjEndIndex
        THETA = (ProjIndex-ProjCenter)*2*PI/ProjScale;
        if (Phantom==1)
          Slice = CreateProjSlice(THETA,U_ScanR,U_HelicP,YL,ZL,U_DW,U_DH);
        else
          Slice = CreateProjSliceDisk(THETA,U_ScanR,U_HelicP,YL,ZL,U_DW,U_DH);
        end
        Slice = Slice*ObjR;
        SliceIndex =ProjIndex-ProjBeginIndex+1; 
        Proj(:,:,SliceIndex)= Slice;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Compute the derivative and filtering
    waitbar(0.5, waitBarHandle, 'Filtering the cone beam projection');
    
    if(DerMode==1)
       if(FilteringMode==1)
         FProj=KatsevichFiltering(Proj,ProjScale,DecWidth,DecHeigh,ScanR,StdDis,HelicP,FilterCoeFF,delta,FilteringMode, keepIntermediateResults);
         FProj1=KatsevichFiltering(Proj,ProjScale,DecWidth,DecHeigh,ScanR,StdDis,HelicP,1-FilterCoeFF,delta,FilteringMode, keepIntermediateResults);
         FProj=(FProj+FProj1)/2;
       else
         FProj=KatsevichFiltering(Proj,ProjScale,DecWidth,DecHeigh,ScanR,StdDis,HelicP,FilterCoeFF,delta,FilteringMode, keepIntermediateResults);
       end       
       ProjCtr = ProjCenter;
       clear FProj1;
    else
       if(FilteringMode==1)
         FProj=KatsevichFiltering_T(Proj,ProjScale,DecWidth,DecHeigh,ScanR,StdDis,HelicP,FilterCoeFF,delta,FilteringMode);
         FProj1=KatsevichFiltering_T(Proj,ProjScale,DecWidth,DecHeigh,ScanR,StdDis,HelicP,1-FilterCoeFF,delta,FilteringMode);
         FProj=(FProj+FProj1)/2;
       else
         FProj=KatsevichFiltering_T(Proj,ProjScale,DecWidth,DecHeigh,ScanR,StdDis,HelicP,FilterCoeFF,delta,FilteringMode);
       end    
       ProjCtr = ProjCenter-0.5;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Backprojection the filtered data
    
    waitbar(0.75, waitBarHandle, 'Backprojection the filtered data');
    geom  = struct(  'Locus',    [ScanR  StdDis HelicP ProjScale ProjBeginIndex ProjEndIndex ProjCtr],  ...
                'Detector',  [DecWidth YL DecHeigh ZL (YL-1)/2 (ZL-1)/2], ...
                'Object',    [ObjR RecSX RecSY RecSZ DeltaX DeltaY DeltaZ]);
    KatsevichBkProj_GPU(geom,FProj,RecMatrix);            
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ProjBeginIndex = ProjEndIndex-1;
    ProjEndIndex   = ProjEndIndex+StepProjNumber-2;
end%%%while
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Time = (cputime-Time)/60;
%save RecResult  RecMatrix ProjNumber ProjScale ProjCenter YL ZL DerMode;




  
    

