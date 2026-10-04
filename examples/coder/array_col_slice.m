function y = array_col_slice()
% Column slice `A(:, j)` is not lowered yet.
A = [1, 2, 3; 4, 5, 6];
y = A(:, 2);
end
