function y = band(x)
    if x >= 10
        y = 100;
    elseif x >= 5
        y = 50;
    elseif x >= 1
        y = 10;
    elseif x >= 0
        y = 1;
    else
        y = -1;
    end
end
