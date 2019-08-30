clc
clear
format long
syms b1 b2;
f = (b1^2 + b2 -11)^2 + (b1 +b2^2 -7)^2;

figure
fcontour(f);
hold on;

for num = 1:10
x = zeros(1,1000);
y = zeros(1,1000);
x(1) = rand;
y(1) = rand;
i = 1; 
stepsize = 0.001;
iteration = 1000;
grad = gradient(f,[b1,b2]);
Gradient = subs(grad ,[b1,b2],[x(1),y(1)]);
while i<iteration
    I = [x(i),y(i)]';
    x(i+1) = I(1) -stepsize*Gradient(1); 
    y(i+1) = I(2) -stepsize*Gradient(2); 
    i = i+1;
    Gradient = subs(grad,[b1,b2],[x(i),y(i)]); 
end
plot(x,y,'*-r');
end
