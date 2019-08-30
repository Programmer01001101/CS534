% Plots for 1b and 1c are all plotted by this single program
 

%close figure windows and clear workspace
close all; clc; warning off;

load HW3_1.mat
% Shuffle the data
Z = [X y];
random_Z = Z(randperm(size(Z, 1)), :);

X = random_Z(:,1:length(X(1,:)));
y = random_Z(:,length(X(1,:))+1);


%Cross Validation starts
%Outer Loop
len = length(X);
testErrorArray = zeros(1,5);
trainingErrorArray = zeros(1,20);
validationError = zeros(1,20);
for i = 1:5
    % Special case for the last fold (has 71 elements)
    if i == 5
        Xtmp = X((1:(i-1)*70),:);
        ytmp = y((1:(i-1)*70),:);
        Xtest =X(((i-1)*70+1:len),:);
        ytest =y(((i-1)*70+1:len),:);
    % Other folds (has 70 elements)
    else
        Xtmp = X([1:(i-1)*70,i*70+1:len],:);
        ytmp = y([1:(i-1)*70,i*70+1:len],:);
        Xtest =X(((i-1)*70+1:i*70),:);
        ytest =y(((i-1)*70+1:i*70),:);
    end
        error =  zeros(4,500);
        %Inner Loop Starts
        for j = 1:4
            len2 = length(Xtmp);
            % Special case for the last fold (has 71 elements)
            if len2 == 281 && j == 4
                Xtrain = Xtmp([1:(j-1)*70,j*70+1:len2],:);
                ytrain = ytmp([1:(j-1)*70,j*70+1:len2],:);
                Xvalid = Xtmp(((j-1)*70+1:j*70),:);
                yvalid = ytmp(((j-1)*70+1:j*70),:);
            else
                Xtrain = Xtmp([1:(j-1)*70,j*70+1:len2],:);
                ytrain = ytmp([1:(j-1)*70,j*70+1:len2],:);
                Xvalid = Xtmp(((j-1)*70+1:j*70),:);
                yvalid = ytmp(((j-1)*70+1:j*70),:);
            end
            
            % Calls Matlab built-in function to calculate the elastic net model.
            B = lassoglm(Xtrain,ytrain,'binomial','Alpha',0.95,'Lambda',lambdas);
            
            
            % Classify the output (if the predicted value is > 0.5, we predict the result as 1, Otherwise as 0)
            ypredict =  Xvalid*B >0.5;
            
            % Calculate error (total wrong predictions / total predictions)
            error(j,:) =  sum(abs(ypredict - (repmat(yvalid,1,500))))./length(yvalid);
            
            
        end

        % Calculate mean error of 4 inner iterations
        errormean = mean(error);
        % standard deviation 
        errorsd = std(error);
        % mean error +- standard error
        errorupper = errormean+errorsd;
        errorlower = errormean-errorsd;
        
        
        % Find lambdamin and lambda*
        [M, I] = min(errormean);
        condition = errormean(I)+ errorsd(I);
        maxIndex = find(errormean <= condition);
        I2 = max(maxIndex);
        
%       lambda*
        lambdaOptimal = lambdas(I2);
        
        % Construct model with best lambda(lambda*) using training folds +
        % validation fold (4 folds total)
        B = lassoglm(Xtmp,ytmp,'binomial','Alpha',0.95,'Lambda',lambdaOptimal);
        % Classify the output (if the predicted value is > 0.5, we predict the result as 1, Otherwise as 0)
        ypredict =  Xtest*B >0.5;
            
        % Calculate test error (total wrong predictions / total predictions)
        testErrorArray(i) =  sum(abs(ypredict - ytest))./length(ytest);
        
        % run inner loop again to calculate errors using lambda =
        % lambdaOptimal
        for j = 1:4
            len2 = length(Xtmp);
            % Special case for the last fold (has 71 elements)
            if len2 == 281 && j == 4
                Xtrain = Xtmp([1:(j-1)*70,j*70+1:len2],:);
                ytrain = ytmp([1:(j-1)*70,j*70+1:len2],:);
                Xvalid = Xtmp(((j-1)*70+1:j*70),:);
                yvalid = ytmp(((j-1)*70+1:j*70),:);
            else
                Xtrain = Xtmp([1:(j-1)*70,j*70+1:len2],:);
                ytrain = ytmp([1:(j-1)*70,j*70+1:len2],:);
                Xvalid = Xtmp(((j-1)*70+1:j*70),:);
                yvalid = ytmp(((j-1)*70+1:j*70),:);
            end
            
            % Calls Matlab built-in function to calculate the model.
            % This time, set lambda = lambdaOptimal
            B = lassoglm(Xtrain,ytrain,'binomial','Alpha',0.95,'Lambda',lambdaOptimal);
            
            
            % Classify the output (if the predicted value is > 0.5, we predict the result as 1, Otherwise as 0)
%             ypredict =  Xvalid*B >0.5;
            
            % Calculate Validation error (total wrong predictions / total predictions)
            validationError((i-1)*4+j)  =  errormean(I);
            
            % Classify the output (if the predicted value is > 0.5, we predict the result as 1, Otherwise as 0)
%             ypredict =  Xtrain*B >0.5;
            
            % Calculate Training error (total wrong predictions / total predictions)
            trainingErrorArray((i-1)*4+1)  =  error(1,I);
            trainingErrorArray((i-1)*4+2)  =  error(2,I);
            trainingErrorArray((i-1)*4+3)  =  error(3,I);
            trainingErrorArray((i-1)*4+4)  =  error(4,I);
            
 
        end
        
        
        
        %Plot Graphs
        logs = log10(lambdas);
        figure
        hold on 
        Error = plot(logs,errormean,'Color','black');
        Sigma = plot(logs,errorlower,'r');
        plot(logs,errorupper,'r')
        
        optimalLambda = plot(logs(I),errormean(I),'bo');
        OptimalLambda2 = plot(logs(I2),errormean(I2),'go');
        plot([-5 2],[errorupper(I) errorupper(I)],'Color','blue','LineStyle','--')
        plot([-5 2],[errorlower(I) errorlower(I)],'Color','blue','LineStyle','--')
        plot([logs(I) logs(I)],[0 1],'Color','blue','LineStyle','--')
        plot([logs(I2) logs(I2)],[0 1],'Color','green','LineStyle','--')
        
        %title, legend, axes
        title('2.b. Model Selection')
        legend([Error Sigma optimalLambda OptimalLambda2 ] , { '{\mu}Error','{\mu}Error +/- {\sigma}Error','{\lambda}min','{\lambda}'''})
        xlabel('log10(\lambda)') 
        ylabel('Classification Eoor') 
        axis([-5 2 0 0.8])
        hold off       
end

% plot Boxplots
group = [ones(size(testErrorArray')); 2 * ones(size(trainingErrorArray'));3 * ones(size(validationError'))];
figure
boxplot([testErrorArray'; trainingErrorArray'; validationError'],group)
set(gca,'XTickLabel',{'testError','trainingError','validationError'})

