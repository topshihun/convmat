function y = continue_sum(n)
    y = 0;
    for i = 1:n
        if i == 2
            continue;
        end
        y = y + i;
    end
end
