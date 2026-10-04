function y = file_io()
    % Read a binary file of doubles via the stdio-backed runtime helpers.
    fid = fopen('data.bin', 'r');
    y = fread(fid);
    fclose(fid);
end
