function sierpinski(maxIter) %#codegen
% Sierpinski triangle, from the MATLAB Coder example gallery.
x = 0;
y = 0;
for k = 1:maxIter
    r = rem(k, 3);
    if r == 0
        x = 0.5 * x;
        y = 0.5 * y;
    elseif r == 1
        x = 0.5 * (x + 1);
        y = 0.5 * y;
    else
        x = 0.5 * x;
        y = 0.5 * (y + 1);
    end
    plot(x, y, '.');
end
end
