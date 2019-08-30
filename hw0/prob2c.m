
clc
clear

syms b1 b2;
f = (b1^2 + b2 -11)^2 + (b1 +b2^2 -7)^2;
fcontour(f, 'Fill', 'On');
hold on;

for num = 1:10
x(1) = rand;
y(1) = rand;
i = 1; 
stepsize = 0.01;
iteration = 1000;
grad = gradient(f,[b1,b2]);
Gradient = subs(grad,[b1,b2],[x(1),y(1)]);
he = hessian(f,[b1,b2]);
hess = subs(inv(he),[b1,b2],[x(1),y(1)]);
 while i<iteration
      z=[x(i),y(i)]-(stepsize*hess*Gradient).' ;
      x(i+1) = z(1);
      y(i+1) = z(2);
      i = i+1;
      Gradient = subs(grad,[b1,b2],[x(i),y(i)]);
      hess = subs(inv(he),[b1,b2],[x(i),y(i)]);
end
plot(x,y,'*-r');
grid on;
end

