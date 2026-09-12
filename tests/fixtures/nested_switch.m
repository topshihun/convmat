function y = nested_switch(n)
    y = 0;
    for i = 1:n
        switch i
            case 1
                y = y + 1;
            case 2
                y = y + 4;
            case 3
                y = y + 9;
            otherwise
                y = y + 0;
        end
    end
end
