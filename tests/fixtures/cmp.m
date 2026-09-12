function y = cmp(a, b)
    y = 0;
    if a == b
        y = y + 1;
    end
    if a ~= b
        y = y + 1;
    end
    if a < b
        y = y + 1;
    end
    if a <= b
        y = y + 1;
    end
    if a > b
        y = y + 1;
    end
    if a >= b
        y = y + 1;
    end
end
