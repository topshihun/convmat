function y = cmp_arith(a, b)
    y = (a > b) + (a < b) + (a == b) + (a >= b) + (a <= b) + (a ~= b);
end
