function t = struct_inout(s)
    t = struct('a', s.a, 'b', s.b);
    t.a = t.a + 1;
end
