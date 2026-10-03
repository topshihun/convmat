function y = anon_copy(a)
    f = @(x) x + a;
    g = f;
    y = g(1);
end
