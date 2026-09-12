function r = binary_math(a, b)
    r = pow(a, b) + atan2(b, a) + hypot(a, b) + mod(a, b) + rem(a, b) + min(a, b) + max(a, b);
end
