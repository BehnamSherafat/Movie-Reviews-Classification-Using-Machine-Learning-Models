function [xForDisplay] = SVMCrossFold()
tic
%% Initial Parameters
max_epoch_cross = 10; % You are free to change maximum epoch number
nFold = 5; % Please do not change cross-fold number
initial_learning_rate = [10^0, 0.5, 0.2, 0.1, 0.01, 0.001]; % % Please do not change learning rate matrix
tradeoff = [10^11, 10^10, 10^9, 10^8, 10^7, 10^6]; % % Please do not change tradeoff matrix
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

r = randi(nFold,size(trainingdataN, 1),1);
U = [r, trainingdataN];
firstColumn = U(:, 1);
accuracies = cell(size(initial_learning_rate, 2)*size(tradeoff, 2), nFold);
for i = 1:nFold    
    for m = 1:size(initial_learning_rate, 2) 
        for u = 1:size(tradeoff, 2) 
            LR = initial_learning_rate(1, m);
            c = tradeoff(1, u);
            A = U(firstColumn == i, :);
            B = U(firstColumn ~= i, :);
            test = A(:, 2:end);
            train = B(:, 2:end);
            train_label = train(:, 1);
            u_labels = unique(train_label);
            n_labels = length(u_labels); % number of labels
            %% Obtaining Parameters
            num_instances = size(train, 1);
            num_features = size(train, 2)-1;
            %% Initializing W
            w_1 = zeros(1, num_features+1);
            %% Training Using the cross validation train files
            time_step = -1;
            epoch = 0;
            while epoch <= max_epoch_cross-1
                epoch = epoch  + 1;
                train = train(randperm(size(train,1)),:);
                for j = 1:num_instances
                    % Defining learning rate
                    time_step = time_step + 1;
                    learning_rate = LR/(1 + time_step);
                    X = [train(j, 2:num_features+1), 1];
                    summ = dot(X, w_1);
                    if train(j,1)*summ <= 1
                        for k = 1:size(w_1, 2)
                            w_1(1, k) = (1-learning_rate)*w_1(1, k) + learning_rate*c*train(j, 1)*X(1, k);
                        end
                    else
                        for k = 1:size(w_1, 2)
                            w_1(1, k) = (1-learning_rate)*w_1(1, k);
                        end
                    end
                end
                fprintf('Epoch number: %d\n',epoch);
                fprintf('Weight vector and b:');
                disp(w_1)
                
            end
            
            % Evaluating Using the "data.test" set
            
            testingDataSize = size(test, 1);
            Classifications = zeros(testingDataSize,2);
            for k=1:testingDataSize
                Classifications(k, 2) = test(k, 1);
                summ = dot([test(k, 2:num_features+1), 1], w_1);
                if summ >= 0
                    Classifications(k, 1) = +1;
                else
                    Classifications(k, 1) = -1;
                end
            end
            
            confMat=zeros(n_labels);
            for f=1:n_labels
                for o=1:n_labels
                    confMat(f,o)=sum(Classifications(:, 2)==u_labels(f) & Classifications(:, 1)==u_labels(o));
                end
            end
            
            a = (confMat(1,1)+confMat(2,2))/(confMat(1,1)+confMat(1,2)+confMat(2,1)+confMat(2,2)); % Accuracy
            fprintf('Accuracy percentage: %d\n',a);
            
            accuracies{(m-1)*6+u, i} = a;

        end
    end
    
end

Average_acc = mean(cell2mat(accuracies), 2);
disp('---------Accuracy Table--------')
results_1 = [num2cell(repelem(initial_learning_rate,6)'), num2cell(repmat(tradeoff',[6 1])),accuracies, num2cell(Average_acc)];
header = {'Learning Rate','Regularization Parameter','Fold1','Fold2','Fold3','Fold4','Fold5','Average Accuracy'};
xForDisplay = [header; results_1];
disp(xForDisplay)
toc

end