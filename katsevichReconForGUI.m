function RecMatrix=katsevichReconForGUI(COE, FilteringMode, scanGeometry, projectionData, reconImage, useGPU, waitBarHandle, app)
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
ProjNumber= projectionData.getTotalProjectionNumber(); % Total Projection Samping number
ScanR = scanGeometry.getSourceToCenterDistance();       % (cm) Scanning radius
StdDis = scanGeometry.getSourceToDetectorDistance();      % (cm) Source to detector distance
HelicP=  scanGeometry.getHelicalPitch();      % (cm) Helical pitch
ObjR  = reconImage.getObjectRadius();       % (cm) Object radius
ProjScale = projectionData.getProjectionNumberPerTurn(); % Number of projection per turn
ProjCenter= (ProjNumber + 1) / 2; % 

FilterCoeFF= COE;

YL = projectionData.getDetResoU();        % sampling number per row
ZL = projectionData.getDetResoV();         % sampling number per column

%DecWidth =107.8;% (cm) Width of detector array
%DecHeigh =39.1; % (cm) Heigh of detector array
StepProjNumber= 3502;% Projections number per intertiative

RecSX         = reconImage.getResoX();% Recconstruction size of the object
RecSY         = reconImage.getResoY();
RecSZ         = reconImage.getResoZ();

DeltaX = reconImage.getPixelSizeX();
DeltaY = reconImage.getPixelSizeY();
DeltaZ = reconImage.getPixelSizeZ();

XCtrIdx = reconImage.getCtrIdxX();
YCtrIdx = reconImage.getCtrIdxY();
ZCtrIdx = reconImage.getCtrIdxZ();


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
Proj = app.getProjectionData.getProjection();
%Proj     = zeros(YL,ZL,StepProjNumber);
RecMatrix= zeros(RecSX,RecSY,RecSZ);
ProjBeginIndex=1;
ProjEndIndex  = StepProjNumber;
while (ProjBeginIndex<ProjNumber)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Compute the cone projection
    
    Proj(:,:,3502) = 0; % This need to be modified;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Compute the derivative and filtering
    waitbar(0.333, waitBarHandle, 'Filtering the cone beam projection');
    
    if(DerMode==1)
       if(FilteringMode==1)
         FProj=KatsevichFiltering(Proj,ProjScale,DecWidth,DecHeigh,ScanR,StdDis,HelicP,FilterCoeFF,delta,FilteringMode, useGPU);
         FProj1=KatsevichFiltering(Proj,ProjScale,DecWidth,DecHeigh,ScanR,StdDis,HelicP,1-FilterCoeFF,delta,FilteringMode, useGPU);
         FProj=(FProj+FProj1)/2;
       else
         FProj=KatsevichFiltering(Proj,ProjScale,DecWidth,DecHeigh,ScanR,StdDis,HelicP,FilterCoeFF,delta,FilteringMode, useGPU);
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
    
    waitbar(0.666, waitBarHandle, 'Backprojection the filtered data');
    geom  = struct(  'Locus',    [ScanR  StdDis HelicP ProjScale ProjBeginIndex ProjEndIndex ProjCtr],  ...
                'Detector',  [DecWidth YL DecHeigh ZL (YL-1)/2 (ZL-1)/2], ...
                'Object',    [ObjR RecSX RecSY RecSZ DeltaX DeltaY DeltaZ]);
    if useGPU
        KatsevichBkProj_GPU(geom,FProj,RecMatrix);            
    else
        KatsevichBkProj(geom,FProj,RecMatrix);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ProjBeginIndex = ProjEndIndex-1;
    ProjEndIndex   = ProjEndIndex+StepProjNumber-2;
end%%%while




  
    

