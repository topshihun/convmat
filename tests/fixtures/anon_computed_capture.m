function y = anon_computed_capture(a)
    k = a * a;
    f = @(x) x + k;
    y = f(1);
end
