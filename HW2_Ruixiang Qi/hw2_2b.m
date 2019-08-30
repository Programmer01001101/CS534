%close figure windows and clear workspace
close all; clc; warning off;

%initialized identity matrix and mu = [0 0]
muIdentity = [0 0];
Identity = [1 0; 0 1];

% real mu and sigma
mu1 = [2 ; 2];
sigma1 = [2 -1 ; -1 1];

mu2 = [0 ; 0];
sigma2 = [1 0.5 ; 0.5 1];

% diagonalize sigma and use its diagonalization to transform samples from
% N(0,I) to N(mu,sigma)
N = 1000;
R = mvnrnd(muIdentity,Identity,N);

%Sample of X1
[V1,D1,VT1] = eig(sigma1);
T1 = V1 * sqrtm(D1);
Sample1 = T1 * R'+mu1;

%Sample of X2
[V2,D2,VT2] = eig(sigma2);
T2 = V2 * sqrtm(D2);
Sample2 = T2 * R'+mu2;


%mean and covariance of each Sample
m1 = mean(Sample1')';
m2 = mean(Sample2')';
s1 = cov(Sample1');
s2 = cov(Sample2');

%define linear discriminant function
pi = 0.5;
%f is the difference of two linear discriminant functions
syms x y
f = @(x,y)[x; y]' * inv(s1) * m1 - 0.5 * m1' * inv(s1) *m1 + log(pi) - ([x; y]' * inv(s2) * m2 - 0.5 * m2' * inv(s2) *m2 + log(pi));

%define bayes decision function
%f2 is the difference of two discriminants
%note that some of the terms are cancelled out: (d/2ln2pi, for instance) 

%Theoretical
syms x y
f2 = @(x,y) -0.5*log(det(sigma1)) - 0.5* ([x; y]-mu1)'*inv(sigma1)*([x ; y]-mu1)  - (-0.5*log((det(sigma2))) - 0.5* ([x; y]-mu2)'*inv(sigma2)*([x;y]-mu2));

%Empirical
syms x y
f3 = @(x,y) -0.5*log(det(s1)) - 0.5* ([x; y]-m1)'*inv(s1)*([x ; y]-m1) - (-0.5*log((det(s2))) - 0.5* ([x; y]-m2)'*inv(s2)*([x;y]-m2));

%plot the sample points
figure
plot(Sample1(1,:),Sample1(2,:),'o')
hold on
plot(Sample2(1,:),Sample2(2,:),'o')

%plot the function when then function is evaluated to be 0 (that is when
%the two discriminant functions equal each other).
%plot the LDA decision boundary (first graph the function's contour, then only display the part when the function evaluates to 0)
fc = fcontour(f);
fc.LevelList = [0 0];

%plot the Theoretical Bayes decision boundary
fc2 = fcontour(f2,'black','LineWidth',1.2);
fc2.LevelList = [0 0];

%plot the Empirical Bayes decision boundary
fc3 = fcontour(f3,'--k','LineWidth',1.2);
fc3.LevelList = [0 0];

%title, legend, axes
title('2.b. Linear discriminant with large sample')
p1 = plot(1,1,'g');
p2 = plot(2,2,'k');
p3 = plot(3,3,'--k');
legend([p1 p2 p3],{'LDA','Bayes(theoretical)','Bayes(estimated)'})
xlabel('x1') 
ylabel('x2') 
axis([-2 7 -3 5])




hold off

