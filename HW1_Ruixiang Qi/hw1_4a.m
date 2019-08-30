%close figure windows and clear workspace
close all; clc; warning off;

load('HW1_4.mat')

%Plot the samples in a scatterplot
figure
scatter(X,Y)
hold on


%Plot of the true model
syms F X1
F = -3.2591*X1^3 + 4.8439*X1^2 + 1.7046*X1 + 1.0685;
fplot(F,'linewidth',2,'color','black')


%Plot of the Estimated model (Polynomial least squares)
newX = zeros(1000,4);
newX(:,1) = X.^3;
newX(:,2) = X.^2;
newX(:,3) = X;
newX(:,4) = 1;
beta = inv(newX'*newX)*newX'*Y;
G = beta(1)*X1^3 + beta(2)*X1^2 + beta(3)*X1 +beta(4);
fplot(G,'linewidth',2,'color','green')


title('4.a. polynomial least squares')
legend('samples','true model','estimated model')
xlabel('x') 
ylabel('y') 
axis([min(X) max(X) min(Y) max(Y)])

hold off


