function y = ops(a, b)
    p = a * b;
    q = a .* b;
    r = a / b;
    s = a .\ b;
    t = a == b;
    u = a <= b;
    v = a & b;
    y = p + q + r + s + t + u + v;
end
