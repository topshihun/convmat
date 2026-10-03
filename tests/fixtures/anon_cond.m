function y = anon_cond(a)
    f = @(x) x + a;
    if f(3) > 4
        y = 1;
    else
        y = 0;
    end
end
