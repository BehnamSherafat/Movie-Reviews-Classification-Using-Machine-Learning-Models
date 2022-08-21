function [classifications] = Classify(tree, attributes, instance)

actual = instance(1, 1);

% Recursion with 3 cases

% Case 1: Current node is labeled 'true'
% So trivially return the classification as 1
if tree(1).value == 1
    classifications = [1, actual];
    return
end

% Case 2: Current node is labeled 'false'
% So trivially return the classification as 0
if tree(1).value == 0
    classifications = [0, actual];
    return
end

% Case 3: Current node is labeled an attribute
% Follow correct branch by looking up index in attributes, and recur

index = ismember(attributes,tree(1).value)==1;
k = find(index);

if instance(1, k) >= tree(3).value
    classifications = Classify(tree(2).A, attributes, instance);
elseif instance(1, k) < tree(3).value
    classifications = Classify(tree(2).B, attributes, instance);
end
      
return
end