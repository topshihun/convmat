function y = array_size(A)
% `size(A, 1)` of a dynamic-shape parameter: the ABI carries an explicit
% `(rows, cols)` shape descriptor because a bare element count is ambiguous.
y = size(A, 1);
end
