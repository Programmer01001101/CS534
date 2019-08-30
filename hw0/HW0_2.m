%close figure windows and clear workspace
close all; clear all; clc; warning off;

%define numer of update iterations, trials
N = 1e5;
trials = 10;

%define problem domain
[beta_1, beta_2] = meshgrid(-5:0.01:5);

%calculate f over for display
f = (beta_1.^2 + beta_2 - 11).^2 + (beta_1 + beta_2.^2 - 7).^2;

%define gradient function
del_f = @(b_1, b_2) [2*b_1 + 4*b_1*(b_1^2 + b_2 - 11) + 2*b_2^2 - 14;...
    - 2*b_2 + 4*b_2*(b_2^2 + b_1 - 7) + 2*b_1^2 - 22];

%define hessian function
hessian_f = @(b_1, b_2) [12*b_1^2 + 4*b_2 - 42, 4*b_1 + 4*b_2;...
                         4*b_1 + 4*b_2, 12*b_2^2 + 4*b_1 - 26];

%define initialization function
initialize = @(x)10 * [rand(1)-0.5; rand(1)-0.5];

%define gradient descent update function
gradient_update = @(b_1, b_2, gamma) [b_1; b_2] - gamma * del_f(b_1, b_2);

%define newton update function
newton_update = @(b_1, b_2, gamma) [b_1; b_2] - gamma * ...
    inv(hessian_f (b_1, b_2)) * ...
    del_f(b_1, b_2);

%define plotting functions
critical = @(s, color, marker)plot(s(1,:), s(2,:), [color marker]);
trajectory = @(beta, color){plot(beta(1,:), beta(2,:), color),...
                            plot(beta(1,1), beta(2,1), [color 'o'])};

%find stationary points
[c_1, c_2] = solve('2*b_1 + 4*b_1*(b_1^2 + b_2 - 11) + 2*b_2^2 - 14',...
    '- 2*b_2 + 4*b_2*(b_2^2 + b_1 - 7) + 2*b_1^2 - 22',...
    'b_1', 'b_2', 'MaxDegree', 4);
c = [vpa(c_1).'; vpa(c_2).'];

%classify stationary points
minima = false(1,size(c,2));
maxima = minima;
saddle = maxima;
for i = 1:size(c, 2)
    eigs = eig(hessian_f(c(1,i), c(2,i)));
    if(prod(eigs) < 0)
       saddle(i) = true; 
    else
        if eigs(1) < 0
            maxima(i) = true;
        else
            minima(i) = true;
        end
    end
end
                     
%display cost function and roots
figure;
contour(beta_1, beta_2, f, 100); axis equal; hold on;
xlabel('\beta_1'); ylabel('\beta_2');
critical(c(:, maxima), 'r', 'p'); %maxima
critical(c(:, saddle), 'k', 'x'); %saddle
critical(c(:, minima), 'b', 's'); %minima

%gradient descent - learning rate 1e-3
for gradient = 1:trials
    beta = nan(2,N);
    beta(:,1) = initialize();
    for i = 1:N-1
        beta(:,i+1) = gradient_update(beta(1,i), beta(2,i), 1e-3);
    end
    trajectory(beta, 'b');
end

%Newton's method - learning rate 1
for newton = 1:trials
    beta = nan(2,N);
    beta(:,1) = initialize();
    for i = 1:N-1
        beta(:,i+1) = newton_update(beta(1,i), beta(2,i), 1e-2);
    end
    trajectory(beta, 'm');
end

%display legend
legend('cost', 'maxima', 'saddle', 'minima');
text(4, 4, 'gradient');
text(4.05, 4, '_____', 'Color', 'blue');
text(4, 4.1, 'Newton');
text(4.05, 4.1, '_____', 'Color', 'm');