
function [Classifications, a] = SVM(initial_learning_rate, max_epoch_cross, trade_off)
tic
max_epoch = max_epoch_cross;
LR = initial_learning_rate;
c = trade_off;
desired_num_feat = 100; % Desired features after PCA (dimension reduction method)
%% Reading Data and store it in tables

format long
fileID1 = fopen('data.train');
C = textscan(fileID1,'%s');
trainingdataN = zeros('double');
row = 0;
for i = 1:size(C{1})
    A = strsplit(C{1}{i},':');
    if size(A, 2) == 1
        row = row + 1;
        column = 1;
        value = str2double(A(1));
    else
        column = str2double(A(1))+1;
        value = str2double(A(2));
    end
    trainingdataN(row, column) = value;
end


format long
fileID1 = fopen('data.test');
C = textscan(fileID1,'%s');
testingdataN = zeros('double');
row = 0;
for i = 1:size(C{1})
    A = strsplit(C{1}{i},':');
    if size(A, 2) == 1
        row = row + 1;
        column = 1;
        value = str2double(A(1));
    else
        column = str2double(A(1))+1;
        value = str2double(A(2));
    end
    testingdataN(row, column) = value;
end


format long
fileID1 = fopen('data.eval.anon');
C = textscan(fileID1,'%s');
evaldata = zeros('double');
row = 0;
for i = 1:size(C{1})
    A = strsplit(C{1}{i},':');
    if size(A, 2) == 1
        row = row + 1;
        column = 1;
        value = str2double(A(1));
    else
        column = str2double(A(1))+1;
        value = str2double(A(2));
    end
    evaldata(row, column) = value;
end

%% Saving data as Sparse data
spaTrain = sparse(trainingdataN);
save('spaTrain.mat','spaTrain');

spaTest = sparse(testingdataN);
save('spaTest.mat','spaTest');

spaEva = sparse(evaldata);
save('spaEva.mat','spaEva');

%% Load Sparse files
load('spaTrain.mat')
load('spaTest.mat')
load('spaEva.mat')

% Making the features' dimension of data same as each other
trainingdataN = spaTrain;
testingdataN = [spaTest, zeros(size(spaTest, 1), 15)];
evaldata = [spaEva, zeros(size(spaEva, 1), 1)];

%% Changing sparse data to full data
trainingdataN = full(trainingdataN);
testingdataN = full(testingdataN);
evaldata = full(evaldata);

%% Implementing PCA to reduce the dimension
PCA_f = desired_num_feat/(size(trainingdataN, 2)-1);
[COEFF, ~, ~] = pca(trainingdataN(:, 2:end)); % Obtaining coefficients matrix
p = round(PCA_f*size(trainingdataN, 2)-1);
Ap = COEFF(:, 1:p);

trainingdataN = [trainingdataN(:, 1), bsxfun(@minus, trainingdataN(:, 2:end), mean(trainingdataN(:, 2:end)))*Ap];
testingdataN = [testingdataN(:, 1), bsxfun(@minus, testingdataN(:, 2:end), mean(trainingdataN(:, 2:end)))*Ap];
evaldataN = [evaldata(:, 1), bsxfun(@minus, evaldata(:, 2:end), mean(trainingdataN(:, 2:end)))*Ap];

%% Save dimension reducted matrices
save('trainingdataN.mat', 'trainingdataN', '-mat');
save('testingdataN.mat', 'testingdataN', '-mat');
save('evaldataN.mat', 'evaldataN', '-mat');

%% Load dimension reducted matrices
load('trainingdataN.mat','-mat');
load('testingdataN.mat','-mat');
load('evaldataN.mat','-mat');
load('data.eval.anon.id');

%% Changing 0 labels to -1
trainingdataN(trainingdataN(:, 1) == 0) = -1;
testingdataN(testingdataN(:, 1) == 0) = -1;
evaldataN(evaldataN(:, 1) == 0) = -1;

train_label = trainingdataN(:, 1);
u_labels = unique(train_label);
n_labels = length(u_labels); % number of labels
%% Obtaining Parameters
num_instances = size(trainingdataN, 1);
num_features = size(trainingdataN, 2)-1;
%% Initializing W
w_1 = zeros(1, num_features+1);    
%% Training Using the cross validation train files
time_step = -1;
epoch = 0;
while epoch <= max_epoch-1
    epoch = epoch  + 1;
    trainingdataN = trainingdataN(randperm(size(trainingdataN,1)),:);
    for j = 1:num_instances
        % Defining learning rate
        time_step = time_step + 1;
        learning_rate = LR/(1 + time_step);
        X = [trainingdataN(j, 2:num_features+1), 1];
        summ = dot(X, w_1);
        if trainingdataN(j,1)*summ <= 1
            for k = 1:size(w_1, 2)
                w_1(1, k) = (1-learning_rate)*w_1(1, k) + learning_rate*c*trainingdataN(j, 1)*X(1, k);  
            end
        else
            for k = 1:size(w_1, 2)
                w_1(1, k) = (1-learning_rate)*w_1(1, k); 
            end
        end
    end
        
    fprintf('Epoch number: %d\n',epoch);
    
    % Evaluating Using the "data.test" set
    
    testingDataSize = size(testingdataN, 1);
    Classifications = zeros(testingDataSize,2);
    for k=1:testingDataSize
        Classifications(k, 2) = testingdataN(k, 1);
        summ = dot([testingdataN(k, 2:num_features+1), 1], w_1);
        if summ >= 0
            Classifications(k, 1) = +1;
        else
            Classifications(k, 1) = -1;
        end
    end
    
    confMat=zeros(n_labels);
    for i=1:n_labels
        for j=1:n_labels
            confMat(i,j)=sum(Classifications(:, 2)==u_labels(i) & Classifications(:, 1)==u_labels(j));
        end
    end
    
    a = (confMat(1,1)+confMat(2,2))/(confMat(1,1)+confMat(1,2)+confMat(2,1)+confMat(2,2)); % Accuracy
    fprintf('Accuracy percentage on the development set: %d\n',a);
    
    results{epoch, 1} = epoch;
    results{epoch, 2} = a;
    results{epoch, 3} = w_1;
    
end

header = {'Epoch','Accuracy Percentage','Average Weight Vector'};
xForDisplay = [header; results];
disp(xForDisplay)
%% Testing Using the cross validation test file
evaDataSize = size(evaldataN, 1);
Classifications = zeros(evaDataSize,2);

W = results{cell2mat(results(:, 2)) == max(cell2mat(results(:, 2))), 3};

for k=1:evaDataSize
    Classifications(k, 2) = evaldataN(k, 1);
    summ = dot([evaldataN(k, 2:num_features+1), 1], W);
    if summ >= 0
        Classifications(k, 1) = +1;
    else
        Classifications(k, 1) = -1;
    end
end

confMat=zeros(n_labels);
for i=1:n_labels
    for j=1:n_labels
        confMat(i,j)=sum(Classifications(:, 2)==u_labels(i) & Classifications(:, 1)==u_labels(j));
    end
end

a = (confMat(1,1)+confMat(2,2))/(confMat(1,1)+confMat(1,2)+confMat(2,1)+confMat(2,2)); % Accuracy

fprintf('Maximum development set accuracy: %d\n',max(cell2mat(results(:, 2))));
fprintf('Test set accuracy using weight vector with maximum accuracy in the development set: %d\n',a);

plot(cell2mat(results(:, 1)), cell2mat(results(:, 2)))
title('Development Set Accuracy vs Epoch',...
    'FontSize', 18, 'FontWeight','bold');
y=ylabel({'Development Set', 'Accuracy (%)'}, 'FontSize', 18, 'FontWeight','bold');
set(y, 'Units', 'Normalized', 'Position', [-0.1, 0.5, 0]);
set(get(gca,'ylabel'),'rotation',0)
xlabel('Epoch', 'FontSize', 10, 'FontWeight','bold');

submission_file = [data_eval_anon(:, 1), Classifications(:, 1)]; % Creating the submission files
submission_file(submission_file(:, 1) == -1) = 0; % Change back the -1 values to 0 values
submission_file;
toc
end