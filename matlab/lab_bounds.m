% define constants 
bin_size = 10;
lab_min = -110;
lab_max =  110;
range = lab_min:bin_size:lab_max;
n = length(range);

l = 50;

write_plots_to_file = 1;

% calculate the boundaries
bins = zeros(n^2, 4);

count = 0;
for j = range
    for i = range
        count = count + 1;
        a_min = i;
        a_max = i + (bin_size - 1);
        b_min = j;
        b_max = j + (bin_size - 1);
        
        bins(count, :) = [a_min, a_max, b_min, b_max];
    end
end

% Create a pixel for each bound
img_l =  ones(n, n) * l;
img_a = zeros(n, n);
img_b = zeros(n, n);

count = 0;
for i = 1:n^2
        count = count + 1;
        a_min = bins(i, 1);
        b_min = bins(i, 3);
        
        img_a(count) = a_min + (bin_size / 2);
        img_b(count) = b_min + (bin_size / 2);
end

img = zeros(n, n, 3);
img(:, :, 1) = img_l;
img(:, :, 2) = img_a;
img(:, :, 3) = img_b;

% get the (in and out-gamut) rgb values
sRGB = lab2rgb(img);
aRGB = lab2rgb(img, 'ColorSpace', 'adobe-rgb-1998');

% set out-gamut values to white-color
sRGB_gamut = gamutfilter(sRGB);
aRGB_gamut = gamutfilter(aRGB);

% display plots
figure(1); imshow(sRGB, 'InitialMagnification', 2000)
figure(2); imshow(sRGB_gamut, 'InitialMagnification', 2000)
figure(3); imshow(aRGB, 'InitialMagnification', 2000)
figure(4); imshow(aRGB_gamut, 'InitialMagnification', 2000)

% if write_plots_to_file
%     imwrite(sRGB, 'sRGB.png', 'png', 'InitialMagnification', 2000)
%     imwrite(sRGB_gamut, 'sRGB_gamut.png', 'png', 'InitialMagnification', 2000)
%     imwrite(aRGB, 'aRGB.png', 'png', 'InitialMagnification', 2000)
%     imwrite(aRGB_gamut, 'aRGB_gamut.png', 'png', 'InitialMagnification', 2000)
% end

% function definition to filter out-gamut
function out = gamutfilter(img)
    dim = size(img);
    n_row = dim(1);
    n_col = dim(2);
    
    if dim(3) ~= 3
      error('not an image');
    end
    
    out = zeros(n_row, n_col, 3);
    for i = 1:n_row
        for j = 1:n_col
            r = img(i, j, 1);
            g = img(i, j, 2);
            b = img(i, j, 3);
            
            if r < 0 || r > 1
                out(i, j, :) = [1, 1, 1];
            elseif g < 0 || g > 1
               out(i, j, :) = [1, 1, 1];
            elseif b < 0 || b > 1
                out(i, j, :) = [1, 1, 1];
            else
               out(i, j, :) = [r, g, b];
            end
        end
    end
end



