% colormap需要再调一下
% python还需要实现一个可以拖动的图像窗口
 
[x, t] = meshgrid(linspace(0, 25, 50) / 24.0,linspace(0, 575.5, 1151) / 575 .* 17 .* pi - 2 .* pi);
p = pi / 2 .* exp(-t / (8 .* pi));
u = 1 - (1 - mod(3.6 .* t, 2 .* pi) / pi) .^ 4 / 2;
y = 2 .* (x .^ 2 - x) .^ 2 .* sin(p);
r = u .* (x .* sin(p) + y .* cos(p));
surf(r .* cos(t), r .* sin(t), u .* (x .* cos(p) - y .* sin(p)))
