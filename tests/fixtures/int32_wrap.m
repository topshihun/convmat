function y = int32_wrap()
% int32 overflow wraps (INT32_MAX + 1 == INT32_MIN).
y = int32(2147483647) + int32(1);
end
