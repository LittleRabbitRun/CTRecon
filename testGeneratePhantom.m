fid = fopen('out.raw');
img = fread(fid,512*512*512,'double');
img = reshape(img,512,512,512);
figure;
for ii = 1: 512
    reimg = img(ii,:,:);
    reimg = reshape(reimg,512,512);
    imshow(reimg,[]);

    pause(0.002);
end