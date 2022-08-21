function [bestfeature, I, currEntropy, gains, avg] = InformationGain(data, features, signFeatures)

numbFeatures = length(signFeatures);

numbT = sum(data(:, 1) == 1);
p = numbT/size(data, 1);

numbF = sum(data(:, 1) == 0);    
n = numbF/size(data, 1);


if p==0
   currEntropy = -n*log2(n);
elseif n==0
   currEntropy = -p*log2(p);
else
   currEntropy = -p*log2(p)-n*log2(n);
end

gains = -1*ones(1, numbFeatures);

for i=2:numbFeatures+1
    if signFeatures(i-1) == 1
        avg = mean(data(:, i));
        H = zeros();
        idx = data(:, i) >= avg;
        f = [sum(idx == 1);sum(idx == 0)];
        D = [];
        M = [];
        q=1;
        w=1;
        for k = 1: size(idx, 1) 
           if idx(k,1 ) == 1
               D(q, :)  = data(k, :);
               q=q+1;
           elseif idx(k,1 ) == 0
               M(w, :)  = data(k, :);
               w=w+1;
           end
        end
        idxp = D(:, 1) == 1;
        numT = sum(idxp);

        idxn = D(:, 1) == 0;
        numF = sum(idxn);

        p = numT/(q-1);
        n = numF/(q-1);
        if p==0
           H(1, 1) = -n*log2(n);
        elseif n==0
           H(1, 1) = -p*log2(p);
        else
           H(1, 1) = -p*log2(p)-n*log2(n);
        end
        
        idxp = M(:, 1) == 1;
        numT = sum(idxp);

        idxn = M(:, 1) == 0;
        numF = sum(idxn);

        p = numT/(w-1);
        n = numF/(w-1);
        if p==0
           H(2, 1) = -n*log2(n);
        elseif n==0
           H(2, 1) = -p*log2(p);
        else
           H(2, 1) = -p*log2(p)-n*log2(n);
        end
        
        EE = dot(f, H)/size(data, 1);
        gains(1, i-1) = currEntropy - EE;
    end
end 

[~,I] = max(gains);
bestfeature = features(1, I+1);

