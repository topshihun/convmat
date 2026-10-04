function y = builtin_clamp(x)
% Clamp a scalar to the [0, 1] range using nested min/max.
y = min(max(x, 0), 1);
end
