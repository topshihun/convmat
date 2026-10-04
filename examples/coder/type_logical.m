function y = type_logical()
% Explicit `logical` conversion is not lowered yet.
y = logical(1) + logical(0);
end
