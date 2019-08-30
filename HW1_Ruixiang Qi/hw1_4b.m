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
fplot(F,'linewidth',3,'color','black')


%Points contain all x,y pairs
Points = zeros(1000,2);
Points(:,1)=X;
Points(:,2)=Y;
count = 0;

ConsensusSet = [];
index=1;

% RANSAC process
while count/length(X)<0.4
    count = 0;
    ConsensusSet = [];
    index=1;
    %Pick 5 random points
    index = randsample(1:length(Points), 5);
    PickedPoints = Points(index,:);
    p = polyfit(PickedPoints(:,1),PickedPoints(:,2),3);

    for i = 1 : length(X)
        x = Points(i,1);
        y = Points(i,2);
        if abs(y-polyval(p,x)) <= 0.3
            ConsensusSet(index,1) = x;
            ConsensusSet(index,2) = y;
            index = index+1;
            count = count +1;
        end
    end
end

% Recalculate Final Model using concensus set.
syms X1
p = polyfit(ConsensusSet(:,1),ConsensusSet(:,2),3);
G = p(1)*X1^3 + p(2)*X1^2 + p(3)*X1 +p(4);
fplot(G,'linewidth',2,'color','green')

%Display Final Consensus Set
scatter(ConsensusSet(:,1),ConsensusSet(:,2),[],[255 153 153]/256)


title('4.b. polynomial least squares')
legend('samples','true model','estimated model','Consensus Set')
xlabel('x') 
ylabel('y') 
axis([min(X) max(X) min(Y) max(Y)])
hold off

