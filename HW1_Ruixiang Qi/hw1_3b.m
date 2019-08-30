%close figure windows and clear workspace
close all; clc; warning off;

load('HW1_3.mat')

%Set learning rate and #of updates, initialize coefficient and deriavative vectors
beta = rand(1,100);
del_f = zeros(1,100);
gamma = 5e-3;
N = 1e4;


%training process
for j = 1:N
    % the gradient of the Loss function
     del_f = -2*X'*(Y-X*beta')/100;
     for i = 1:100
        beta(i) = beta(i) - gamma*del_f(i);
     end
end
%Plot the coefficients
stem(beta);



title('3.b. model coeficients W/O regularazation')
xlabel('model weight index') 
ylabel('model weight') 
hold off