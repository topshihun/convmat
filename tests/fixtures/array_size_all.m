function y = array_size_all(A)
% `size(A)` returns the `[rows cols]` vector; sum its entries.
s = size(A);
y = s(1) + s(2);
end
