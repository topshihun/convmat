function y = array_row_slice()
% Row slice `A(i, :)` is not lowered yet.
A = [1, 2, 3; 4, 5, 6];
y = A(1, :);
end
