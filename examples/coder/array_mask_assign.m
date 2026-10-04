function y = array_mask_assign()
% Logical-mask assignment `A(A < 0) = 0` is not lowered yet.
A = [1, -2, 3, -4];
A(A < 0) = 0;
y = A;
end
