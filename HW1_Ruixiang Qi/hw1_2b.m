%close figure windows and clear workspace
close all; clc; warning off;

%initialized identity matrix and mu = [0 0]
muIdentity = [0 0];
Identity = [1 0; 0 1];

% real mu and sigma
mu = [1 ; 1];
sigma = [1 -0.5 ; -0.5 0.5];

% diagonalize sigma and use its diagonalization to transform samples from
% N(0,I) to N(mu,sigma)
[V,D,VT] = eig(sigma);
R = mvnrnd(muIdentity,Identity,1000);
T = V * sqrtm(D);
Sample = T * R'+mu;
d = sqrt(diag(D));


% Plot random samples
figure
plot(Sample(1,:),Sample(2,:),'.')
hold on


%Plot level curves of PDF
x1 = -3:.1:4; x2 = -1:.1:3;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu',sigma);
F = reshape(F,length(x2),length(x1));
contour(x1,x2,F,'LineWidth',1);
 
%Plot EigenVectors
quiver(1,1,V(1,2),V(2,2),d(2),'r','LineWidth',1);
quiver(1,1,V(1,1),V(2,1),d(1),'r','LineWidth',1);


title('2.b. samples w/ pdf')
legend('samples','pdf','eigenvector1','eigenvector2')
xlabel('x1') 
ylabel('x2') 
axis([-3 4 -1 3])

hold off