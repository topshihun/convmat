function y = type_integer()
% Integer types (`int32`, ...) are not lowered yet.
y = int32(3) + int32(4);
end
