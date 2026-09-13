function y = index_end()
    a = [1, 2; 3, 4];
    y = a(end) + a(end, 1);
end
