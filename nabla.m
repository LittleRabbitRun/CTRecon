% Calcuate the nabla_x or nabla_y or nabla_xy = sqrt(nabla_x^2 + nabla_y^2)
% f : the 2D image
% delta: size of the pixel (monosize in x/y direction)
% direction: 'x', 'y' or 'xy' to calculate the directional gradient of an image

function [nabla_f] = CTNabla(f, delta, direction)
[numL, numC] = size(f);
assert(strcmp(direction, 'x') || strcmp(direction, 'y') || ...
    strcmp(direction, 'xy'), 'The third argument must be ''x'', ''y'' or ''xy''');
if strcmp(direction, 'x')
    f_plus1 = f(:,2:end);
    f_plus1 = [f_plus1, zeros(numL,1)];
    nabla_f = (f_plus1 - f) ./ delta;
elseif strcmp(direction, 'y')
    f_plus1 = f(2:end,:);
    f_plus1 = [f_plus1; zeros(1, numC)];
    nabla_f = (f_plus1 - f) ./ delta;
else
    nablax = nabla(f, delta, 'x');
    nablay = nabla(f, delta, 'y');
    nabla_f = sqrt(nablax.^2 + nablay.^2);
end

end
