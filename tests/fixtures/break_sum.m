function y = break_sum(n)
    y = 0;
    for i = 1:n
        if i > 3
            break;
        end
        y = y + i;
    end
end
