function y = anon_snapshot(a)
    f = @(x) x + a;
    a = 100;
    y = f(1);
end
