%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     CT/Micro CT lab
%     Department of Radiology
%     University of Iowa
%     Version of 2003.03.05
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the unit impulse response of the hilbert transform
% XS represents the length which has a form of 2^n
% Index represents the window function type
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [HS]=CreateHSP(XS,index)
Length=XS;
HS=ones(Length,1);
Center=(Length)/2+1;
PI=3.14159265358979;
HS(1)=0;
for i=2:Center-1
    HS(i)=2*sin(PI*(i-Center)/2)^2/(PI*(i-Center));
end
HS(Center)=0;
for i=Center+1:Length
    HS(i)=2*sin(PI*(i-Center)/2)^2/(PI*(i-Center));
end

switch (index)
    case 1
        %rectangle window
        Window=ones(Length,1);
    case 2
        %kaiser window
        Window=kaiser(Length,2.5);
    case 3
        %hamming window
        Window=Hamming(Length)';
    case 4
        %hanning window
        Window=Hann(Length)';
    case 5
        %blackman window
        Window=BlackMan(Length)';
    otherwise
        Window=ones(Length,1);
end
HS=HS.*Window;
end


function w = Hamming(L)
PI=3.14159265358979;
    N = L - 1;
    w = 0.54 - 0.46 * cos(2*PI*(0:N) / N);
end

function w = Hann(L)
PI=3.14159265358979;
    N = L - 1;
    w = 0.5 * (1 - cos(2 * PI * (0:N) / N));
end

function w = BlackMan(N)
PI=3.14159265358979;
    if mod(N,2)
        M = (N + 1) / 2;
    else
        M = N / 2;
    end
    w = 0.42 - 0.5 * cos(2 * PI * (0:M-1) / (N-1)) + 0.08 * cos(4 * PI * (0:M-1) / (N-1));
    wf = w(end:-1:1);
    w = [w, wf];
end






