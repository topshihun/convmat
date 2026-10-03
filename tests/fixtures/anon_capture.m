function y = anon_capture(a)
    f = @(x) x * x + a;
    y = f(3);
end
