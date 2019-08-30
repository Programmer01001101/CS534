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
     % the gradient of the Loss function without the sub-gradient term
     subL1 = -2*X'*(Y-X*beta');
     %update the deriavative vector with the sub-gradient term
     for i = 1:100
        if beta(i)> 1
             del_f(i) = (subL1(i) + beta(i) -1)/100 ;
        elseif beta(i) < -1
             del_f(i) = (subL1(i) + beta(i) + 1)/100;
        else
             del_f(i) = subL1(i)/100;
        end
     end
     % update coefficients
     beta = beta - gamma*del_f;
end
%Plot the coefficients
stem(beta);


title('3.a. LASSO model coeficients')
xlabel('model weight index') 
ylabel('model weight') 
hold off

% We may see that the resulting coefficients are 11, 30,34,85,95


