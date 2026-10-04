function y = char_switch(c)
% Switching on a char compares code points.
switch c
    case 'a'
        y = 1;
    case 'b'
        y = 2;
    otherwise
        y = 0;
end
end
