%% Q3
clc
clear
% The information of the Pyramid
r = 2;
h = 4;
n = 6;
t = linspace(0,2*pi,n+1);
t = t(1:end-1);
xbase = r*cos(t(1:end)).';
ybase = r*sin(t(1:end)).';
zbase = zeros(size(xbase));
V = [xbase ybase zbase
    0 0 h];

f_base = 1:n;
f_triangle = zeros(n,n);
for k = 1:n-1
    f_triangle(k,1:3) = [k,k+1,n+1];
end
f_triangle(n,1:3) = [n,1,n+1];
f_triangle(:,4:end) = NaN;

F = [f_base
    f_triangle];

% Draw the Pyramid
figure
hold on
patch('Vertices',V,'Faces',F,'FaceColor','interp','FaceVertexCData',hsv(n+1),'FaceAlpha',0.5,'EdgeColor','b')
plot3([0 0],[0 0],[0 h],'r:','LineWidth',2)
for q = 1:3
plot3([xbase(q) xbase(q+3)],[ybase(q) ybase(q+3)],[0 0],':r','LineWidth',2)
end
text(0,0,h/3,"\it h=4cm")
text(xbase(1)/2,1,0,"\it r=2cm")
hold off

% Set the axis
axis equal
axis off
view(3)

title("DSC2409005:Final Question 3",'FontName','Comic Sans MS')
