function y = while_break(n)
    y = 0;
    i = 0;
    while i < n
        i = i + 1;
        if i > 3
            break;
        end
        y = y + i;
    end
end
