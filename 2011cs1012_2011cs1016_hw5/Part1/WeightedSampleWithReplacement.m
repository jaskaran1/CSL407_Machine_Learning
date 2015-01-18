function [ indices ] = WeightedSampleWithReplacement( D,n )
%Function for weighted sampling
% if sum(D)~=1
%     fprintf('Sum of probabilities isnt equal to 1\n');
% end
cumsum=zeros(1,n);
indices=zeros(1,n);
tot=0;
%create cumulative sum array
for i=1:n
    tot=tot+D(i);
    cumsum(i)=tot;
end
for i=1:n
    prob=rand(1);
    for j=1:n
        if prob<=cumsum(j)
            indices(i)= j;
            break;
        end
    end
end

end

