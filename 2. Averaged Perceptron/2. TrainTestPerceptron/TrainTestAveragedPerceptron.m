function [submission_file, num_updates, results] = TrainTestAveragedPerceptron(learning_rate, max_epoch)

%% Defining Initial Parameters

desired_num_feat = 100; % Desired features after PCA (dimension reduction method)

%% Reading Data and store it in tables

format long
fileID1 = fopen('data.train');
C = textscan(fileID1,'%s');
trainingdata = zeros('double');
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
    trainingdata(row, column) = value;
end


format long
fileID1 = fopen('data.test');
C = textscan(fileID1,'%s');
testingdata = zeros('double');
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
    testingdata(row, column) = value;
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
spaTrain = sparse(trainingdata);
save('spaTrain.mat','spaTrain');

spaTest = sparse(testingdata);
save('spaTest.mat','spaTest');

spaEva = sparse(evaldata);
save('spaEva.mat','spaEva');

%% Load Sparse files
load('spaTrain.mat')
load('spaTest.mat')
load('spaEva.mat')

% Making the features' dimension of data same as each other
trainingdata = spaTrain;
testingdata = [spaTest, zeros(size(spaTest, 1), 15)];
evaldata = [spaEva, zeros(size(spaEva, 1), 1)];

%% Changing sparse data to full data
trainingdata = full(trainingdata);
testingdata = full(testingdata);
evaldata = full(evaldata);

%% Implementing PCA to reduce the dimension
PCA_f = desired_num_feat/(size(trainingdata, 2)-1);
[COEFF, ~, ~] = pca(trainingdata(:, 2:end)); % Obtaining coefficients matrix
p = round(PCA_f*size(trainingdata, 2)-1);
Ap = COEFF(:, 1:p);

trainingdataN = [trainingdata(:, 1), bsxfun(@minus, trainingdata(:, 2:end), mean(trainingdata(:, 2:end)))*Ap];
testingdataN = [testingdata(:, 1), bsxfun(@minus, testingdata(:, 2:end), mean(trainingdata(:, 2:end)))*Ap];
evaldataN = [evaldata(:, 1), bsxfun(@minus, evaldata(:, 2:end), mean(trainingdata(:, 2:end)))*Ap];

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

%% Obtaining Parameters

num_instances = size(trainingdataN, 1);
num_features = size(trainingdataN, 2)-1;

%% Randomize W and b Vectors

b = -0.01 + (0.01+0.01)*rand(1, 1);
w = -0.01 + (0.01+0.01)*rand(1, num_features);
w_1 = [w, b];

%% Training Using the "data.train" set

a = w_1;
num_updates = 0;
epoch = 0;
while epoch <= max_epoch-1
    epoch = epoch  + 1;
    trainingdataN = trainingdataN(randperm(size(trainingdataN,1)),:);
    for j = 1:num_instances        
        X = [trainingdataN(j, 2:num_features+1), 1];
        sum = dot(X, w_1);
        if sum >= 0
            output(j, 1) = +1;
        else
            output(j, 1) = -1;
        end
        if  trainingdataN(j, 1)*output(j, 1) < 0            
            num_updates = num_updates + 1;
            for k = 1:size(w_1, 2)
                w_1(1, k) = w_1(1, k) + learning_rate*(trainingdataN(j, 1)*X(1, k));  
            end
        end        
         a = a + w_1;
    end    
    a = a /(num_instances);
    fprintf('Epoch number: %d\n',epoch);
    
    % Evaluating Using the "data.test" set

    evaDataSize = size(testingdataN, 1);
    Classifications = zeros(evaDataSize,2);
    numCorrect = 0;
    for k=1:evaDataSize
        Classifications(k, 2) = testingdataN(k, 1);
        sum = dot([testingdataN(k, 2:num_features+1), 1], a);
        if sum >= 0
            Classifications(k, 1) = +1;
        else
            Classifications(k, 1) = -1;
        end
        if isequal(Classifications(k,1), Classifications(k, 2)) %correct
            numCorrect = numCorrect + 1;
        end
    end
    
    % Checking the classification accuracy

    if (evaDataSize)
        Percentage = round(100 * numCorrect / evaDataSize);
    else
        Percentage = 0;
    end
    fprintf('Accuracy percentage on the development set: %d\n',Percentage);    
    results{epoch, 1} = epoch;
    results{epoch, 2} = Percentage;
    results{epoch, 3} = a;
    end

header = {'Epoch','Accuracy Percentage','Average Weight Vector'};
xForDisplay = [header; results];
disp(xForDisplay)

% Testing Using the "data.eva.anon" set
evaDataSize = size(evaldataN, 1);
Classifications = zeros(evaDataSize,2);
numCorrect = 0;
a = results{cell2mat(results(:, 2)) == max(cell2mat(results(:, 2))), 3};

for k=1:evaDataSize
    Classifications(k, 2) = evaldataN(k, 1);
    sum = dot([evaldataN(k, 2:num_features+1), 1], a);
    if sum >= 0
        Classifications(k, 1) = +1;
    else
        Classifications(k, 1) = -1;
    end
    if isequal(Classifications(k,1), Classifications(k, 2)) %correct
        numCorrect = numCorrect + 1;
    end
end

% Checking the classification accuracy

if (evaDataSize)
    Percentage = round(100 * numCorrect / evaDataSize);
else
    Percentage = 0;
end

fprintf('Maximum development set accuracy: %d\n',max(cell2mat(results(:, 2))));
fprintf('Test set accuracy using weight vector with maximum accuracy in the development set: %d\n',Percentage);
fprintf('The total number of updates the learning algorithm performs on the training data: %d\n',num_updates);

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

end