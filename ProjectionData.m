classdef ProjectionData < matlab.mixin.Copyable
    properties
        SourceToCenterDistance = []
        SourceToDetectorDistance = []
        
        DetResoU = []
        DetResoV = []
        DetPixelSizeU = []
        DetPixelSizeV = []
        DetCntIdxU = []
        DetCntIdxV = []

        IsArcDetector = []
        
        StartSourceAngle = []
        StartSourceZPos = []
        Pitch = []
        
        TotalProjectionNumber = []
        ProjectionNumberPerTurn = [] 
        
        projection = []
        StorageOrder = [] % Default Order UVP
    
        sourcePosX = []
        sourcePosY = []
        sourcePosZ = []
        
        xds = []
        yds = []
        zds = []
        
        viewangles = []
        zshifts = []
    end
    
    methods
        function obj = ProjectionData(~)
            obj.SourceToCenterDistance = 538;
            obj.SourceToDetectorDistance = 946;
            
            obj.DetResoU = 888;
            obj.DetResoV = 64;
            obj.DetPixelSizeU = 1.0239;
            obj.DetPixelSizeV = 1.0963;
            
            obj.DetCntIdxU = 443.75; % start Idx: 0 based
            obj.DetCntIdxV = 31.5; % start Idx: 0 based
            obj.IsArcDetector = true;
            obj.StartSourceAngle = 0; % UNIT: Degree
            obj.StartSourceZPos = 0;
            obj.Pitch = 0;
            
            obj.TotalProjectionNumber = 984;
            obj.ProjectionNumberPerTurn = 984;
            
            obj.projection = zeros(obj.DetResoU, obj.DetResoV, obj.TotalProjectionNumber);
            
            obj.StorageOrder = [1, 2, 3]; % U V P order
            
            obj.sourcePosX = 0;
            obj.sourcePosY = obj.SourceToCenterDistance;
            obj.sourcePosZ = 0;
            obj.sourcePosY = obj.y0Callback();
            obj.viewangles = obj.viewAnglesCallBack();
            [obj.xds, obj.yds, obj.zds] = obj.xyzDsCallBack();
            obj.zshifts = obj.zshiftCallBack();
        end
        
        function res = getSubProjectionSet(obj, osNum, osIdx)
            res = ProjectionData;
            views = obj.getViewAngles();
            zsfts = obj.getZShifts();
            
            res.setStartSourceAngle(views(osIdx));
            res.setStartZPos(zsfts(osIdx));
            viewsIdx = osIdx:osNum:obj.getTotalProjectionNumber();
            totalNumProj = numel(viewsIdx);
            res.setTotalProjectionNumber(totalNumProj);
            res.setProjectionNumPerTurn(floor(obj.getProjectionNumberPerTurn() / osNum));
            proj = obj.getProjection();
            res.setProjection(proj(:,:,viewsIdx));
            res.setStorageOrder(obj.getStorageOrder());

            res.setViewAngles(views(viewsIdx));
            res.setZShifts(zsfts(viewsIdx));            
        end
        function x0 = getSourceX(obj)
            x0 = obj.sourcePosX;
        end
        function y0 = getSourceY(obj)
            y0 = obj.sourcePosY;
        end
        function z0 = getSourceZ(obj)
            z0 = obj.sourcePosZ;
        end
        function xds = getXDS(obj)
            xds = obj.xds;
        end
        function yds = getYDS(obj)
            yds = obj.yds;
        end
        function zds = getZDS(obj)
            zds = obj.zds;
        end
        function viewangles = getViewAngles(obj)
            viewangles = obj.viewangles;
        end
        function zshifts = getZShifts(obj)
            zshifts = obj.zshifts;
        end
        
        function WriteFile(obj, fileName)
            fid = fopen(fileName, 'wb');
            fwrite(fid, obj.SourceToCenterDistance, 'double');
            fwrite(fid, obj.SourceToDetectorDistance, 'double');
            fwrite(fid, obj.DetResoU, 'double');
            fwrite(fid, obj.DetResoV, 'double');
            fwrite(fid, obj.DetPixelSizeU, 'double');
            fwrite(fid, obj.DetPixelSizeV, 'double');
            
            fwrite(fid, obj.DetCntIdxU, 'double');
            fwrite(fid, obj.DetCntIdxV, 'double');
            fwrite(fid, obj.IsArcDetector, 'double');
            fwrite(fid, obj.StartSourceAngle, 'double');
            fwrite(fid, obj.StartSourceZPos, 'double');
            fwrite(fid, obj.Pitch, 'double');
            
            fwrite(fid, obj.TotalProjectionNumber, 'double');
            fwrite(fid, obj.ProjectionNumberPerTurn, 'double');
            fwrite(fid, obj.projection, 'double');
            fwrite(fid, obj.StorageOrder, 'double');
            fclose(fid);
        end
        
        function setProjection(obj, prj)
            obj.projection = prj;
        end
        
        function ReadFile(obj, fileName)
            fid = fopen(fileName);
            obj.SourceToCenterDistance = fread(fid, 1, 'double');
            obj.SourceToDetectorDistance = fread(fid, 1, 'double');
            obj.DetResoU = fread(fid, 1, 'double');
            obj.DetResoV = fread(fid, 1, 'double');
            obj.DetPixelSizeU = fread(fid, 1, 'double');
            obj.DetPixelSizeV = fread(fid, 1, 'double');
            obj.DetCntIdxU = fread(fid, 1, 'double');
            obj.DetCntIdxV = fread(fid, 1, 'double');
            obj.IsArcDetector = fread(fid, 1, 'double');
            obj.StartSourceAngle =  fread(fid, 1, 'double');
            obj.StartSourceZPos = fread(fid, 1, 'double');
            obj.Pitch = fread(fid, 1, 'double');
            
            obj.TotalProjectionNumber = fread(fid, 1, 'double');
            obj.ProjectionNumberPerTurn = fread(fid, 1, 'double');
            obj.projection = fread(fid, obj.DetResoU * obj.DetResoV * obj.TotalProjectionNumber, 'double');
            obj.StorageOrder = fread(fid, 3, 'double')';
            if isequal(obj.StorageOrder, [1, 2, 3])
                obj.projection = reshape(obj.projection, obj.DetResoU, obj.DetResoV, obj.TotalProjectionNumber);
            elseif isequal(obj.StorageOrder, [2, 1, 3])
                obj.projection = reshape(obj.projection, obj.DetResoV, obj.DetResoU, obj.TotalProjectionNumber);
            elseif isequal(obj.StorageOrder, [3, 1, 2])
                obj.projection = reshape(obj.projection, obj.TotalProjectionNumber, obj.DetResoU, obj.DetResoV);
            elseif isequal(obj.StorageOrder, [1, 3, 2])
                obj.projection = reshape(obj.projection, obj.DetResoU, obj.TotalProjectionNumber, obj.DetResoV);
            elseif isequal(obj.StorageOrder, [2, 3, 1])
                obj.projection = reshape(obj.projection, obj.DetResoV, obj.TotalProjectionNumber, obj.DetResoU);
            elseif isequal(obj.StorageOrder, [3, 2, 1])
                obj.projection = reshape(obj.projection, obj.TotalProjectionNumber, obj.DetResoV, obj.DetResoU);
            end
            obj.sourcePosY = obj.y0Callback();
            obj.viewangles = obj.viewAnglesCallBack();
            [obj.xds, obj.yds, obj.zds] = obj.xyzDsCallBack();
            obj.zshifts = obj.zshiftCallBack();
        end

        function ord = getStorageOrder(obj)
            ord = obj.StorageOrder;
        end
        
        function setStorageOrder(obj, ord)
            obj.StorageOrder = ord;
        end
        
        function setSourceToDetectorDistance(obj, v)
            obj.SourceToDetectorDistance = v;
        end
        
        function pitch = getPitch(obj)
            pitch = obj.Pitch;
        end
        function setPitch(obj, p)
            obj.Pitch = p;
            obj.sourcePosY = obj.y0Callback();
            obj.viewangles = obj.viewAnglesCallBack();
            [obj.xds, obj.yds, obj.zds] = obj.xyzDsCallBack();
            obj.zshifts = obj.zshiftCallBack();
        end
        
        function startSourceAngle = getStartSourceAngle(obj)
            startSourceAngle = obj.StartSourceAngle;
        end
        function setStartSourceAngle(obj, sa)
            obj.StartSourceAngle = sa;
        end
        
        function startZPos = getStartZPos(obj)
            startZPos = obj.StartSourceZPos;
        end
        function setStartZPos(obj, startZ)
            obj.StartSourceZPos = startZ;
        end
        
        function s2o = getSourceToCenterDistance(obj)
            s2o = obj.SourceToCenterDistance;
        end
        function setSourceToCenterDistance(obj, v)
            obj.SourceToCenterDistance = v;
        end
        
        function s2d = getSourceToDetectorDistance(obj)
            s2d = obj.SourceToDetectorDistance;
        end
        
        function detResoU = getDetResoU(obj)
            detResoU = obj.DetResoU;
        end
        
        function detResoV = getDetResoV(obj)
            detResoV = obj.DetResoV;
        end
        
        function totalProjectionNumber = getTotalProjectionNumber(obj)
            totalProjectionNumber = obj.TotalProjectionNumber;
        end
        
        function projectionNumberPerTurn = getProjectionNumberPerTurn(obj)
            projectionNumberPerTurn = obj.ProjectionNumberPerTurn;
        end
        
        function projectionD = getProjection(obj)
            projectionD = obj.projection;
        end
        
        function setDetResoU(obj, resou)
            if resou ~= obj.DetResoU
                obj.DetResoU = resou;
                obj.projection = ones(obj.DetResoU, obj.DetResoV, obj.TotalProjectionNumber);
                obj.StorageOrder = [1,2,3];  
            obj.sourcePosY = obj.y0Callback();
            obj.viewangles = obj.viewAnglesCallBack();
            [obj.xds, obj.yds, obj.zds] = obj.xyzDsCallBack();
            obj.zshifts = obj.zshiftCallBack();
            end            
        end
        
        function setDetResoV(obj, resov)
            if resov ~= obj.DetResoV
                obj.DetResoV = resov;
                obj.projection = ones(obj.DetResoU, obj.DetResoV, obj.TotalProjectionNumber);
                obj.StorageOrder = [1,2,3];                
            obj.sourcePosY = obj.y0Callback();
            obj.viewangles = obj.viewAnglesCallBack();
            [obj.xds, obj.yds, obj.zds] = obj.xyzDsCallBack();
            obj.zshifts = obj.zshiftCallBack();
            end            
        end
        
        function setTotalProjectionNumber(obj, n)
            if n ~= obj.TotalProjectionNumber
                obj.TotalProjectionNumber = n;
                obj.projection = ones(obj.DetResoU, obj.DetResoV, obj.TotalProjectionNumber);
                obj.StorageOrder = [1,2,3]; 
            obj.sourcePosY = obj.y0Callback();
            obj.viewangles = obj.viewAnglesCallBack();
            [obj.xds, obj.yds, obj.zds] = obj.xyzDsCallBack();
            obj.zshifts = obj.zshiftCallBack();
            end            
        end
        
        function setProjectionNumPerTurn(obj, n)
            obj.ProjectionNumberPerTurn = n;
            obj.sourcePosY = obj.y0Callback();
            obj.viewangles = obj.viewAnglesCallBack();
            [obj.xds, obj.yds, obj.zds] = obj.xyzDsCallBack();
            obj.zshifts = obj.zshiftCallBack();
        end
        
        function u = getDetSizeU(obj)
            u = obj.DetPixelSizeU;
        end
        function setDetSizeU(obj, u)
            obj.DetPixelSizeU = u;
            obj.sourcePosY = obj.y0Callback();
            obj.viewangles = obj.viewAnglesCallBack();
            [obj.xds, obj.yds, obj.zds] = obj.xyzDsCallBack();
            obj.zshifts = obj.zshiftCallBack();
        end
        
        function v = getDetSizeV(obj)
            v = obj.DetPixelSizeV;
        end        
        function setDetSizeV(obj, v)
            obj.DetPixelSizeV = v;
            obj.sourcePosY = obj.y0Callback();
            obj.viewangles = obj.viewAnglesCallBack();
            [obj.xds, obj.yds, obj.zds] = obj.xyzDsCallBack();
            obj.zshifts = obj.zshiftCallBack();
        end
        
        function uCntIdx = getDetCntIdxU(obj)
            uCntIdx = obj.DetCntIdxU;
        end
        function setDetCntIdxU(obj, u)
            obj.DetCntIdxU = u;
            obj.sourcePosY = obj.y0Callback();
            obj.viewangles = obj.viewAnglesCallBack();
            [obj.xds, obj.yds, obj.zds] = obj.xyzDsCallBack();
            obj.zshifts = obj.zshiftCallBack();
        end
        
        function vCntIdx = getDetCntIdxV(obj)
            vCntIdx = obj.DetCntIdxV;
        end
        function setDetCntIdxV(obj, v)
            obj.DetCntIdxV = v;
            obj.sourcePosY = obj.y0Callback();
            obj.viewangles = obj.viewAnglesCallBack();
            [obj.xds, obj.yds, obj.zds] = obj.xyzDsCallBack();
            obj.zshifts = obj.zshiftCallBack();
        end
        
        function isArcDet = getIsArcDetector(obj)
            isArcDet = obj.IsArcDetector;
        end
        function setIsArcDetector(obj, val)
            obj.IsArcDetector = val;
            obj.sourcePosY = obj.y0Callback();
            obj.viewangles = obj.viewAnglesCallBack();
            [obj.xds, obj.yds, obj.zds] = obj.xyzDsCallBack();
            obj.zshifts = obj.zshiftCallBack();
        end       
   
            
 
        function setViewAngles(obj, views)
            obj.viewangles = views;
        end
        function setZShifts(obj, zshft)
            obj.zshifts = zshft;
        end
        
        function y0 = y0Callback(obj)
            y0 = obj.SourceToCenterDistance;
        end
        
        function ang = viewAnglesCallBack(obj)
            ang = (0:(obj.TotalProjectionNumber - 1)) * 2 * pi ...
                / obj.ProjectionNumberPerTurn + obj.StartSourceAngle / 180 * pi;
        end
        
        function [x,y,z] = xyzDsCallBack(obj)
            if obj.IsArcDetector
                alphas=((1:obj.DetResoU) - (obj.DetCntIdxU + 1)) * ...
                    obj.DetPixelSizeU / obj.SourceToDetectorDistance;
                x = sin(alphas) * obj.SourceToDetectorDistance;
                y = obj.SourceToCenterDistance - obj.SourceToDetectorDistance * cos(alphas);
            else
                x = ((1:obj.DetResoU) - (obj.DetCntIdxU + 1)) * ...
                    obj.DetPixelSizeU;
                y = (obj.SourceToCenterDistance - obj.SourceToDetectorDistance) * ...
                    ones(size(obj.xds));
            end
            z = ((1:obj.DetResoV)- obj.DetCntIdxV - 1) * obj.DetPixelSizeV;            
        end
        
        function zs = zshiftCallBack(obj)
            zs = (obj.viewangles - obj.StartSourceAngle / 180 * pi) ... 
                / (2 * pi) * obj.Pitch + obj.StartSourceZPos;
        end
   end
end