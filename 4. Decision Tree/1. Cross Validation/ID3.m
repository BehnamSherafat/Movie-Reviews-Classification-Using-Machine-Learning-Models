function [tree] = ID3(data, features, signFeatures, depth, maxDepth)
    
if (isempty(data))
    error('No examples');
end
%% Define Parameters
numbFeatures = length(signFeatures);

%% Implementing ID3 Algorithm
% Create the tree node
tree = struct('value', 'null', 'A', {'null','null'}, 'B', {'null','null'});
% If all data labels are the same 

if numel(unique(data(:, 1)))==1
    tree(1).value = data(1, 1);    
    fields = fieldnames(tree(1));
    for i = 1:numel(fields)
        if strcmp(tree(1).(fields{i}), 'null')            
            tree = rmfield(tree,fields{i});
        end
    end    
    tree(2).value = depth;  
    return
end
    
if numel(unique(data(:, 1))) ~= 1 && numbFeatures == 0        
    numT = sum(data(:, 1) == 1);
    numF = sum(data(:, 1) == 0);    
    if numT>numF
        tree(1).value = 1;
    else
        tree(1).value = 0;     
    end    
    tree(2).value = depth;
    return
end


[bestfeature, numbFeature, currEntropy, gains, avg] = InformationGain(data, features, signFeatures);

tree(1).value = bestfeature;

if depth >= maxDepth
    
    numT = sum(data(:, 1) == 1);

    numF = sum(data(:, 1) == 0);
    
    if numT>numF
        tree(1).value = 1;
    else
        tree(1).value = 0;     
    end
    
    tree(2).value = depth;
    
    return
else
    tree(2).value = depth+1;
    tree(3).value = avg;
end

signFeatures(numbFeature) = 0;
tree(1).A = '>= Average';
tree(1).B = '< Average';

data_N1  = data(data(:, numbFeature+1) >= avg, :);
% calculating Most Common Value
if isempty(data_N1)
    leaf = struct('value', 'null', 'A', {'null','null'}, 'B', {'null','null'});
    numT = sum(data(:, 1) == 1);
    numF = sum(data(:, 1) == 0);
    if numT>numF
        leaf(1).value = 1;
    else
        leaf(1).value = 0;     
    end            
    tree(2).A = leaf;    
    leaf(2).value = depth;
else        
    
    tree(2).A = ID3(data_N1, features, signFeatures, tree(2).value, maxDepth);
    
end

data_N2  = data(data(:, numbFeature+1) < avg, :);
% calculating Most Common Value
if isempty(data_N2)
    leaf = struct('value', 'null', 'A', {'null','null'}, 'B', {'null','null'});
    numT = sum(data(:, 1) == 1);
    numF = sum(data(:, 1) == 0);
    if numT>numF
        leaf(1).value = 1;
    else
        leaf(1).value = 0;     
    end        
    tree(2).B = leaf;
    leaf(2).value = depth;
else        
    tree(2).B = ID3(data_N2, features, signFeatures, tree(2).value, maxDepth);       
end

return
end
