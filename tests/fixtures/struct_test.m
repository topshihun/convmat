function y = struct_test()
    s = struct('a', 1, 'b', 2);
    s.a = 10;
    y = s.a + s.b;
end
