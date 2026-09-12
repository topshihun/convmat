function y = nested(n)
    y = 0;
    for i = 1:n
        if i > 2
            j = i;
            while j > 0
                y = y + 1;
                j = j - 1;
            end
        end
    end
end
