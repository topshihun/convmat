function y = shape_intro()
    y = size([1, 2; 3, 4], 1) + size([1, 2; 3, 4], 2) + numel([1, 2; 3, 4]) + length([1, 2; 3, 4]);
end
