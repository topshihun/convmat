function y = clamp_hi(x, hi)
    y = x;
    if x > hi
        y = hi;
    end
end
