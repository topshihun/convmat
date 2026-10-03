function y = anon_reassign(a)
    f = @(x) x + a;
    f = @(x) x - a;
    y = f(1);
end
