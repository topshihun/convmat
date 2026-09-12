function [r, t] = polar(x, y)
    r = x * x + y * y;
    t = x / y;
end
