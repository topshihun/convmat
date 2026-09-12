function y = unary_mix(a, b)
    y = -(a + b) + (+a) - (~(a > b)) + a';
end
