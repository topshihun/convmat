function y = anon_multi_capture(a, b)
    f = @(x) x + a * b;
    y = f(2);
end
