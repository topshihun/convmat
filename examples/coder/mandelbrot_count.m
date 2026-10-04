function count = mandelbrot_count(c, maxIterations) %#codegen
% Iteration count for the Mandelbrot set, from the MATLAB Coder example
% gallery. `c` is complex in the original; here it is a real scalar.
z = 0;
count = maxIterations;
for n = 0:maxIterations - 1
    if abs(z) > 2
        count = n;
        break
    end
    z = z^2 + c;
end
end
