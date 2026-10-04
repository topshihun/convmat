function y = text_switch(c)
% Switching on a character value is not lowered yet.
switch c
    case 'a'
        y = 1;
    case 'b'
        y = 2;
    otherwise
        y = 0;
end
end
