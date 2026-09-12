function y = bounded_sum(a, b)
    y = 0;
    while a > 0 && b > 0
        y = y + a + b;
        a = a - 1;
        b = b - 2;
    end
end
