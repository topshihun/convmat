function y = logic_mix(a, b, c)
    p = (a > 0) & (b > 0);
    q = (a < 0) | (b < 0);
    r = (a > 0 && b > 0) || (c > 0);
    y = p + q + r + ~(a == b);
end
