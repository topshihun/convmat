function y = struct_nested()
% Nested struct fields (`s.a.b`) are not lowered yet.
s.a.b = 1;
y = s.a.b;
end
