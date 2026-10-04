function y = linalg_det3()
% det of a 3x3 whose first pivot is zero (forces a row swap).
A = [0, 2, 1; 3, 1, 4; 1, 5, 2];
y = det(A);
end
