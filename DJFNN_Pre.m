% DJFNN prediction
% inputs: model learned by DJFNN and inputs of test samples: x_test
% output: Y_pre: predictions at inputs x_test 
%         B:     N*C, activision of jth inputs for ith rule, 

function [B,Y_pre] = DJFNN_Pre ( model, X_test)

N = size(X_test,1);
C = model.C;
Fuzzi = model.Fuzzi;
Q = model.Q;
A = model.A;
scale = model.scale;
intercept = model.intercept;
order = model.Order;
dim = model.dim;

M = zeros(1,dim);
Par = cell(1,dim);
lb = zeros(1,dim);ub = lb;
for i =  1:dim
    M(i) = Fuzzi.input{i}.MFsNum;
    Par{i} = Fuzzi.input{i}.MFsPar;
    lb(1,i) = Fuzzi.input{i}.MFsPar(1);
    ub(1,i) = Fuzzi.input{i}.MFsPar(end);
end

% normalization
X_test = (X_test - repmat(intercept,N,1))./repmat(scale,N,1);

% check wether X_test is within the definition of fuzzy sets
indicator = all (min(X_test) >= lb) && all (max(X_test) <= ub);
if ~indicator
    disp('the prediction dominon is not within the definition of FM');
    return
end  

% define T(S)-norm
T_norm = @( a, b ) a .* b;
% S_norm = @( a, b ) a + b - a*b;
% T_norm = @( a, b ) min( a , b);
% S_norm = @( a, b ) max( a , b );
% T_norm = @( a, b ) b.*(a==1 & b~=1) + a.*(b==1 & a~=1) + 1.*(a==1 & b==1) + 0.*(a~=1 & b~=1);
S_norm = @( a, b ) b.*(a==0) + a.*(b==0) + 1.*(a~=0 & b~=0);

% B matrix
% tic
% B = zeros(N,C);
% for i = 1 : C
%     s = zeros(N,dim);
%     for l = 1 : dim
%         for j = 1 : M(l)
%             a = trapmf(X_test(:,l),Par{l}(j,:));
%             s_temp = T_norm(a,Q{l}(j,i));
% %             s_temp = a.*Q{l}(j,i);
%             sq = S_norm(s_temp,s(:,l));
% %             sq = s(:,l).*(s_temp==0) + s_temp.*(s(:,l)==0) + 1.*(s_temp~=0 & s(:,l)~=0);
%             s(:,l) = sq;
%         end
%     end
%     s_temp = ones(N,1);
%     for l = 1 : dim           
%         B(:,i) = T_norm(s_temp,s(:,l));
% %         B(:,i) = s_temp.*s(:,l);
%         s_temp = B(:,i);
%     end
% end
% toc

% B matrix (the faster way)
s = zeros(N,C,dim);
for l = 1 : dim
    for j = 1 : M(l)
        a = trapmf(X_test(:,l),Par{l}(j,:));
        s_temp = T_norm(a,Q{l}(j,:));
        sq = S_norm(s_temp,s(:,:,l));
        s(:,:,l) = sq;
    end
end
s_temp = ones(N,1);
for l = 1 : dim           
    B = T_norm(s_temp,s(:,:,l));
    s_temp = B;
end


% D matrix linear
if order == 0
    D = 1./sum(B,2).*B;
else
    if order == 1
        x_temp = [ones(N,1),X_test];
    elseif order == 2
        x_temp = [ones(N,1),X_test,X_test.*X_test,X_test(:,1).*X_test(:,2)];
    elseif order == 3
        x_temp = [ones(N,1),X_test,X_test.*X_test,X_test(:,1).*X_test(:,2),...
                 X_test.^3,X_test(:,1).^2.*X_test(:,2),X_test(:,2).^2.*X_test(:,1)];
    else
        disp('the order of consequence is larger than 3')
        return
    end      
    p = size(x_temp,2);
    D = ones(N,p*C);
    for l = 1 : p*C
        f = mod(l,p);
        if f == 0
            f = p;
            e = l/p;
        else
            e = (l-f)/p + 1;
        end
        D(:,l) = 1./sum(B,2).*B(:,e).*x_temp(:,f);
    end
end 

Y_pre = D*A;

end



