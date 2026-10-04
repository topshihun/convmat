function y = sys_file_io()
% File I/O is not lowered yet.
fid = fopen('data.bin', 'r');
y = fread(fid);
fclose(fid);
end
