function y = anon_loop(n)
    f = @(x) x * 2;
    y = 0;
    for i = 1:n
        y = y + f(i);
    end
end
