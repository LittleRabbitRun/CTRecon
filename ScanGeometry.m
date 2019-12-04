classdef ScanGeometry
    properties
        SourceToCenterDistance = []
        SourceToDetectorDistance = []
    end
    
    methods
        function obj = ScanGeometry(obj)
            obj.SourceToCenterDistance = 538;
            obj.SourceToDetectorDistance = 946;
        end
        
        function readFile(obj, fileName)
            fid = fopen(fileName);
            obj.SourceToCenterDistance = fread(fid, 1, 'double');
            obj.SourceToDetectorDistance = fread(fid, 1, 'double');
            fclose(fid);
        end
        
        function writeFile(obj, fileName)
            fid = fopen(fileName, 'wb');
            fwrite(fid, obj.SourceToCenterDistance, 'double');
            fwrite(fid, obj.SourceToDetectorDistance, 'double');
            fclose(fid);
        end
        
        function s2o = getSourceToCenterDistance(obj)
            s2o = obj.SourceToCenterDistance;
        end
        
        function s2d = getSourceToDetectorDistance(obj)
            s2d = obj.SourceToDetectorDistance;
        end
        
        function setSourceToCenterDistance(obj, s2o)
            obj.SourceToCenterDistance = s2o;
        end
        
        function setSourceToDetectorDistance(obj, s2d)
            obj.SourceToDetectorDistance = s2d;
        end
    end
end