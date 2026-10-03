function y = anon_expr(a)
    f = @(x) x + a;
    g = @(x) x * a;
    y = f(3) * g(4) + f(1);
end
