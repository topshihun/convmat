function y = array_logical_index()
% Logical indexing `A(A > 0)` produces a runtime-sized result; not lowered.
A = [1, -2, 3, -4];
y = A(A > 0);
end
