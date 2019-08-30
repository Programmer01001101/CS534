%close figure windows and clear workspace
close all; clc; warning off;

load('HW1_3.mat')

%Set learning rate and #of updates, initialize coefficient and deriavative
%vectors. beta and del_f are for LASSO method ;  beta2 and del_f2 are for
%ordinary method.
beta = rand(1,100);
beta2 = rand(1,100);
del_f = zeros(1,100);
del_f2 = zeros(1,100);
gamma = 5e-3;
N = 1e4;

%training process
for j = 1:N
     % the gradient of the Loss function without the sub-gradient term
     subL1 = -2*X'*(Y-X*beta');
     for i = 1:100
         %update the deriavative vector with the sub-gradient term for the
         %gradient for LASSO method
        if beta(i)> 1
             del_f(i) = subL1(i) + beta(i) -1 ;
        elseif beta(i) < -1
             del_f(i) = subL1(i) + beta(i) + 1;
        else
             del_f(i) = subL1(i);
        end
         % the gradient for ordinary method remains the same 
        del_f2(i) = subL1(i);
     end
     % update coefficients
     beta = beta - gamma*del_f/100;
     beta2 = beta2 - gamma*del_f2/100;
end


% Calculate the esitimated data
dataLasso = X_test*beta';
dataOrdinary = X_test*beta2';

% Calculate the  mean-square error
errLasso = immse(dataLasso,Y_test);
errOrdinary = immse(dataOrdinary,Y_test);

% Display the  mean-square error
errLasso
errOrdinary



