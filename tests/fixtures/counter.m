function y = counter(n)
    persistent count;
    count = count + n;
    y = count;
end
