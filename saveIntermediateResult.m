function saveIntermediateResult(name, data)
    [xx, yy, zz] = size(data);
    fid = fopen([name,'_', num2str(xx),'x', num2str(yy),'x', num2str(zz),'.data'], 'wb');
    fwrite(fid, data, 'double');
    fclose(fid);
end