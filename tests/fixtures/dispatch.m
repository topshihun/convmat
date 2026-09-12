function y = dispatch(x)
    switch x * 2
        case 2
            y = 10;
        case 4
            y = 20;
        case 6
            y = 30;
        otherwise
            y = 0;
    end
end
