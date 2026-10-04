function y = mask_assign()
% Masked in-place assignment `A(A < 0) = 0` zeroes the negative entries.
A = [1, -2, 3, -4];
A(A < 0) = 0;
y = A;
end
