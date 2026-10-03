function y = zz_nested(a, n)
    y = 0;
    for i = 1:n
        h = @(x) x + a;
        y = y + h(i);
    end
end
