function y = dynamic_ops(A, j)
    % Exercise the dynamic-array primitives: runtime `Inf` fill, runtime
    % subscript write, runtime column slice, and elementwise `min`.
    n = size(A, 1);
    y = Inf(1, n);
    y(j) = 0;
    y = min(y, A(:, j));
end
