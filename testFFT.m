
ConvRes33 = reshape(ConvRes,ProjNumber*AL,YL);
ConvRes33(ProjNumber*AL,nn2) = 0;
for Aindex=1:size(ConvRes33,1)
    TempData = double(ConvRes(Aindex,:));
    FFT_S = fft(TempData(:),nn2);
    TempData = real(ifft(FFT_S.*FFT_F));
    
    ConvRes33(Aindex,:) = TempData;
    %disp(Aindex);
end
ConvRes33 = reshape(ConvRes33,[ProjNumber,AL,nn2]);