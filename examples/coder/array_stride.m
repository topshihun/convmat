function y = array_stride()
% Strided range indexing `A(1:2:end)` is not lowered yet.
A = [1, 2, 3, 4, 5];
y = A(1:2:5);
end
