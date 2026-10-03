function y = anon_lin(a)
    f = @(x, t) x * t + a;
    y = f(3, 4);
end
