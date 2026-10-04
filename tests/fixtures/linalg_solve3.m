function y = linalg_solve3()
% Solve a 3x3 linear system with a unique solution.
A = [2, 1, 0; 1, 3, 1; 0, 1, 2];
b = [1; 2; 3];
y = A \ b;
end
