function y = builtin_binary_math(a, b)
% Binary math builtins that lower to libm calls: atan2, hypot, mod, rem.
y = atan2(a, b) + hypot(a, b) + mod(a, b) + rem(a, b);
end
