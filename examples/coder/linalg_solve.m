function y = linalg_solve()
% Linear solve `A \ b` for a square `A` (runtime helper).
A = [2, 0; 0, 2];
b = [2; 4];
y = A \ b;
end
