% Behnam Sherafat
% Project
% Machine Learning
% Random Forest

function[Classifications, tree, votes, Percentage] = RandomForest(nT, maxDepth)

format long
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

%% Training Section

evalDataSize = size(evaldataN, 1);     
initialDepth = 0;
votes = zeros(size(evaldataN, 1), nT);
for t = 1:nT
    % Draw T bootstrap samples of data with replacement
    numb_rows = round(sqrt(size(trainingdataN, 1)));
    numb_columns = round(sqrt(size(trainingdataN, 2)));

    sampled_data = datasample(trainingdataN,numb_rows); % Draw bootstrap samples of data with replacement
%     sampled_data = [BT_sampled_data(:, 1), datasample(BT_sampled_data(:,2:end),numb_columns, 2,'Replace', false)]; % Draw sample of available features at each split without replacement
    traFeatures = [0, 1:(size(sampled_data, 2)-1)];
    remainedFeatures = traFeatures(1:length(traFeatures));
    signFeatures = ones(1, length(traFeatures) - 1);  
    [tree] = ID3(sampled_data, traFeatures, signFeatures, initialDepth, maxDepth);
    Classifications = zeros(evalDataSize,2);
    numCorrect = 0;
    for k=1:evalDataSize
        try
            Classifications(k,:) = Classify(tree, remainedFeatures, evaldataN(k,:));
            votes(k, t) = Classifications(k,1);
        catch
        % Nothing to do
        end
    end   
end 
for k=1:evalDataSize
        try
        Classifications(k,1) = mode(votes(k, :));
        if isequal(Classifications(k,1), Classifications(k, 2)) %correct
            numCorrect = numCorrect + 1;
        end
        catch
        % Nothing to do
        end
end            
if (evalDataSize)
    Percentage = round(100 * numCorrect / evalDataSize);
else
    Percentage = 0;
end            
Percentage
end    