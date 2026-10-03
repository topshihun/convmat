function y = anon_two(a)
    f = @(x) x + a;
    g = @(x) x * a;
    y = f(3) + g(4);
end
