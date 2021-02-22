% This MATLAB code visualizes the dump files of code.cu and makes an animated gif

clc
clear
close all

list = 0:500:18500; % modify this according to the number of dump files

for i = 1 : length(list)
    name = arrayfun(@(i)['solution_iter_' num2str(i) '.txt'],list(i),'un',0);
    delimiterIn = ' ';
    array{i} = importdata(char(name), delimiterIn);
end
    
h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'solution.gif';
for n = 1:length(list)
    resolution = 15;
    [M,c] = contour(array{n}, resolution,'--','ShowText','on'); 
    c.LineWidth = 3;
    axis equal
    set(gca,'color','w');
    set(gca,'Visible','on')
    xlabel X
    ylabel Y
    drawnow 
      % Capture the plot as an image 
      frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if n == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
      end 
 end