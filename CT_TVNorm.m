% Calculate the TV norm of an image
% f: 2d Image
% delta : pixel size (mono along x/y direction)
function tvNorm = CT_TVNorm(f, delta)
nabla_f = CTNabla(f, delta, 'xy');
tvNorm = sum(sum(nabla_f));
end
