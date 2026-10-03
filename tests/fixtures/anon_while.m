function y = anon_while(n)
    f = @(x) x + 1;
    y = 0;
    while y < n
        y = f(y);
    end
end
