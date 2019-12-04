%% Reconstruction image class
classdef ReconImage < matlab.mixin.Copyable
    properties
        resoX = []
        resoY = []
        resoZ = []
        
        pixelSizeX = []
        pixelSizeY = []
        pixelSizeZ = []
        
     %   objRadius = []
        
        ctrIdxX = []
        ctrIdxY = []
        ctrIdxZ = []
        
        imgCenterX = []
        imgCenterY = []
        imgCenterZ = []
        
        image = [] % The Image itself
        
        storageOrder = []
    end
    
    methods
        
        function obj = ReconImage(obj)
            obj.resoX = 512;
            obj.resoY = 512;
            obj.resoZ = 64;
            
            obj.pixelSizeX = 0.9766;
            obj.pixelSizeY = 0.9766;
            obj.pixelSizeZ = 0.9766;
            
        %    obj.objRadius = 250;
            
            obj.ctrIdxX = (obj.resoX - 1) / 2;
            obj.ctrIdxY = (obj.resoY - 1) / 2;
            obj.ctrIdxZ = (obj.resoZ - 1) / 2;
            
            obj.imgCenterX = 0;
            obj.imgCenterY = 0;
            obj.imgCenterZ = 0;
            
            obj.image = zeros(obj.resoX, obj.resoY, obj.resoZ);
            obj.storageOrder = [1,2,3];
        end
        
        function img = getImage(obj)
            img = obj.image;
        end
        
        function setImage(obj, img)
            obj.image = img;
        end
        
        function WriteFile(obj, fileName)
            fid = fopen(fileName, 'wb');
            fwrite(fid, obj.resoX, 'double');
            fwrite(fid, obj.resoY, 'double');
            fwrite(fid, obj.resoZ, 'double');
            
            fwrite(fid, obj.pixelSizeX, 'double');
            fwrite(fid, obj.pixelSizeY, 'double');
            fwrite(fid, obj.pixelSizeZ, 'double');
            
            fwrite(fid, obj.ctrIdxX, 'double');
            fwrite(fid, obj.ctrIdxY, 'double');
            fwrite(fid, obj.ctrIdxZ, 'double');
            
            fwrite(fid, obj.imgCenterX, 'double');
            fwrite(fid, obj.imgCenterY, 'double');
            fwrite(fid, obj.imgCenterZ, 'double');
            
            fwrite(fid, obj.image, 'double');            
            fwrite(fid, obj.storageOrder, 'double');
            
            fclose(fid);
        end
           
        function ReadFile(obj, fileName)
            fid = fopen(fileName);
            obj.resoX = fread(fid, 1, 'double');
            obj.resoY = fread(fid, 1, 'double');
            obj.resoZ = fread(fid, 1, 'double');
            
            obj.pixelSizeX = fread(fid, 1, 'double');
            obj.pixelSizeY = fread(fid, 1, 'double');
            obj.pixelSizeZ = fread(fid, 1, 'double');
            
            obj.ctrIdxX = fread(fid, 1, 'double');
            obj.ctrIdxY = fread(fid, 1, 'double');
            obj.ctrIdxZ = fread(fid, 1, 'double');
            
            obj.imgCenterX = fread(fid, 1, 'double');
            obj.imgCenterY = fread(fid, 1, 'double');
            obj.imgCenterZ = fread(fid, 1, 'double');
            
            obj.image = fread(fid, obj.resoX * obj.resoY * obj.resoZ, 'double');
            obj.storageOrder = fread(fid, 3, 'double')';
            if isequal(obj.storageOrder, [1,2,3])
                obj.image = reshape(obj.image, obj.resoX, obj.resoY, obj.resoZ);
            elseif isequal(obj.storageOrder, [2,1,3])
                obj.image = reshape(obj.image, obj.resoY, obj.resoX, obj.resoZ);
            elseif isequal(obj.storageOrder, [1,3,2])
                obj.image = reshape(obj.image, obj.resoX, obj.resoZ, obj.resoY);
            elseif isequal(obj.storageOrder, [2,3,1])
                obj.image = reshape(obj.image, obj.resoY, obj.resoZ, obj.resoX);
            elseif isequal(obj.storageOrder, [3,1,2])
                obj.image = reshape(obj.image, obj.resoZ, obj.resoX, obj.resoY);
            elseif isequal(obj.storageOrder, [3,2,1])
                obj.image = reshape(obj.image, obj.resoZ, obj.resoY, obj.resoX);
            end
            fclose(fid);
        end
        
        
        function setResoX(obj, v)
            if(v ~= obj.resoX)
                obj.resoX = v;
                obj.ctrIdxX = (obj.resoX - 1) / 2 - obj.imgCenterX / obj.pixelSizeX;
                obj.image = ones(obj.resoX, obj.resoY, obj.resoZ);
                obj.storageOrder = [1,2,3];
            end            
        end
        
        function setResoY(obj, v)
            if(v~=obj.resoY)
                obj.resoY = v;
                obj.ctrIdxY = (obj.resoY - 1) / 2 - obj.imgCenterY / obj.pixelSizeY;
                obj.image = ones(obj.resoX, obj.resoY, obj.resoZ);
                obj.storageOrder = [1,2,3];
            end
        end
        function setResoZ(obj, v)
            if(v~=obj.resoZ)
                obj.resoZ = v;
                obj.ctrIdxZ = (obj.resoZ - 1) / 2 - obj.imgCenterZ / obj.pixelSizeZ;
                obj.image = ones(obj.resoX, obj.resoY, obj.resoZ);
                obj.storageOrder = [1,2,3];
            end
        end
        
        function setPixelSizeX(obj, v)
            obj.pixelSizeX = v;
            obj.ctrIdxX = (obj.resoX - 1) / 2 - obj.imgCenterX / obj.pixelSizeX;
        end
        function setPixelSizeY(obj, v)
            obj.pixelSizeY = v;
            obj.ctrIdxY = (obj.resoY - 1) / 2 - obj.imgCenterY / obj.pixelSizeY;
        end
        function setPixelSizeZ(obj, v)
            obj.pixelSizeZ = v;
            obj.ctrIdxZ = (obj.resoZ - 1) / 2 - obj.imgCenterZ / obj.pixelSizeZ;
        end
        
        function setCtrIdxX(obj, v)
            obj.ctrIdxX = v;
            obj.imgCenterX = ((obj.resoX - 1) / 2 - obj.ctrIdxX) * obj.pixelSizeX;
        end
        
        function setCtrIdxY(obj, v)
            obj.ctrIdxY = v;
            obj.imgCenterY = ((obj.resoY - 1) / 2 - obj.ctrIdxY) * obj.pixelSizeY; 
        end
        function setCtrIdxZ(obj, v)
            obj.ctrIdxZ = v;
            obj.imgCenterZ = ((obj.resoZ - 1) / 2 - obj.ctrIdxZ) * obj.pixelSizeZ;
        end

        function setImgCenterX(obj, v)
            obj.imgCenterX = v;
            obj.ctrIdxX = (obj.resoX - 1) / 2 - obj.imgCenterX / obj.pixelSizeX;
        end
        
        function setImgCenterY(obj, v)
            obj.imgCenterY = v;
            obj.ctrIdxY = (obj.resoY - 1) / 2 - obj.imgCenterY / obj.pixelSizeY;
        end
        
        function setImgCenterZ(obj, v)
            obj.imgCenterZ = v;
            obj.ctrIdxZ = (obj.resoZ - 1) / 2 - obj.imgCenterZ / obj.pixelSizeZ;
        end
        
        function imgCtrX = getImgCenterX(obj)
            imgCtrX = obj.imgCenterX;
        end
        function imgCtrY = getImgCenterY(obj)
            imgCtrY = obj.imgCenterY;
        end
        function imgCtrZ = getImgCenterZ(obj)
            imgCtrZ = obj.imgCenterZ;
        end
        
        function resX = getResoX(obj)
            resX = obj.resoX;
        end
        
        function resY = getResoY(obj)
            resY = obj.resoY;
        end
        
        function resZ = getResoZ(obj)
            resZ = obj.resoZ;
        end
        
        function pixelSizeX = getPixelSizeX(obj)
            pixelSizeX = obj.pixelSizeX;
        end
        
        function pixelSizeY = getPixelSizeY(obj)
            pixelSizeY = obj.pixelSizeY;
        end
        
        function pixelSizeZ = getPixelSizeZ(obj)
            pixelSizeZ = obj.pixelSizeZ;
        end
        
        function ctrIdxX = getCtrIdxX(obj)
            ctrIdxX = obj.ctrIdxX;
        end
        
        function ctrIdxY = getCtrIdxY(obj)
            ctrIdxY = obj.ctrIdxY;
        end
        
        function ctrIdxZ = getCtrIdxZ(obj)
            ctrIdxZ = obj.ctrIdxZ;
        end
    end
end
