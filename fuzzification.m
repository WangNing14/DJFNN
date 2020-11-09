% fuzzification of inputs
% x_obs: inputs of training samples;
% MFs_num: the number of membership functions for each input
% method = 0: uniform-Width, method = 1:uniform-Frequency
% fuzzi: structure type, serve as input of DJFNN

function fuzzi = fuzzification( x_obs,MFs_num,method )

p = size( x_obs , 2 );
fuzzi.input_number = p;
fuzzi.input = cell(1,p);

for i = 1 : p
    
    fuzzi.input{i}.MFsType = 'trapmf'; 
    
    data = x_obs(:,i);  
    if method == 0
        Par = linspace(min(data),max(data),MFs_num(i));
    elseif method == 1
        per = 0:100/(MFs_num(i)-1):100;
        Par = prctile(data,per);
        Par = unique(Par);       
    else
        disp('error in method setting')
        return
    end
    fuzzi.input{i}.MFsNum = length(Par);
    
    d = 3*(Par(2) - Par(1)); % extend the definition MF to more wide range
    if length(Par) == 2 % case of 2 MFs
        fuzzi.input{i}.MFsPar = [Par(1)-d,Par(1)-d,Par(1),Par(2);...
                                 Par(1),Par(2),Par(2)+d,Par(2)+d];
    else % case of more than 2 MFs
        fuzzi.input{i}.MFsPar(1,:) = [Par(1)-d,Par(1)-d,Par(1),Par(2)];
        for j = 2 :length(Par) - 1
            fuzzi.input{i}.MFsPar(j,:) = [Par(j-1),Par(j),Par(j),Par(j+1)];
        end
        fuzzi.input{i}.MFsPar(length(Par),:) = [Par(end-1),Par(end),Par(end)+d,Par(end)+d];
    end
     
%     figure(i)
%     scatter(data,zeros(N,1))
%     hold on
%     xx = linspace(min(data),max(data),1000);
%     for j = 1 : fuzzi.input{i}.MFsNum
%         yy = trapmf(xx,fuzzi.input{i}.MFsPar(j,:));
%         plot(xx,yy)
%     end   
end

num0 = 0;
for i = 1 : p
    num = num0 + size(fuzzi.input{i}.MFsPar,1);
    num0 = num;
end
fuzzi.mf_number = num;

end

    
    
        
        
        
        
        
        
    