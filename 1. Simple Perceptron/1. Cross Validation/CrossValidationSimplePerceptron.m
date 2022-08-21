function [xForDisplay] = CrossValidationSimplePerceptron()

%% Defining Initial Parameters

cross_fold = 5; % Please do not change cross-fold number
max_epoch_cross = 10; % You are free to change maximum epoch number
learning_rate = [1, 0.5, 0.2, 0.1, 0.01, 0.001]; % You are free to add other values to learning rate matrix
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

%% Training Section

% Creating cross-folds
r = randi(10,size(trainingdataN, 1),1);
U = [r, trainingdataN];
firstColumn = U(:, 1);

results = cell(size(learning_rate, 2), cross_fold); 

for i = 1:cross_fold
    
    for m = 1:size(learning_rate, 2) 
        
        LR = learning_rate(1, m);
        A = U(firstColumn == i, :);
        B = U(firstColumn ~= i, :);
        test = A(:, 2:end);
        train = B(:, 2:end);
               
        %% Obtaining Parameters

        num_instances = size(train, 1);
        num_features = size(train, 2)-1;

        %% Randomize W and b Vectors

        b = -0.01 + (0.01+0.01)*rand(1, 1);
        w = -0.01 + (0.01+0.01)*rand(1, num_features);
        w_1 = [w, b];

        %% Training Using the cross validation train files

        epoch = 0;
        while epoch <= max_epoch_cross-1
            epoch = epoch  + 1;
            train = train(randperm(size(train,1)),:);
            for j = 1:num_instances
                X = [train(j, 2:num_features+1), 1];
                sum = dot(X, w_1);
                if sum >= 0
                    output(j, 1) = +1;
                else
                    output(j, 1) = -1;
                end
                if  train(j, 1)*output(j, 1) < 0
                    for k = 1:size(w_1, 2)
                        w_1(1, k) = w_1(1, k) + LR*(train(j, 1)*X(1, k));  
                    end
                end
            end
            fprintf('Epoch number: %d\n',epoch);
            fprintf('Weight vector and b:');
            disp(w_1)
        end

        %% Testing Using the cross validation test file

        testingDataSize = size(test, 1);
        Classifications = zeros(testingDataSize,2);
        numCorrect = 0;
        for k=1:testingDataSize
            Classifications(k, 2) = test(k, 1);
            sum = dot([test(k, 2:num_features+1), 1], w_1);
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
        if (testingDataSize)
            Percentage = round(100 * numCorrect / testingDataSize);
        else
            Percentage = 0;
        end
        fprintf('Accuracy percentage: %d\n',Percentage);        
        results{m, i} = Percentage;
    end
end

Average = mean(cell2mat(results), 2);
results = [num2cell(learning_rate'),results, num2cell(Average)];
header = {'Learning Rate','Fold1','Fold2','Fold3','Fold4','Fold5','Average'};
xForDisplay = [header; results];
disp(xForDisplay)

end



