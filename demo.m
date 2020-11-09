% a demo to show how to train the DJFNN model based on training samples (DJFNN.M)
% and how to predict outputs at new inputs (DJFNN_Pre.m)
% the example is wizmir dataset, which has 9 inputs, 1 output and 1609
% samples, with five fold cross validation data

clc;clear;dbstop if error;

%% load the training and test data 
% the user can load their own data matrix with column represents inputs-output and 
% row represents each training or test samples
data_tr = importdata('wankara-5-5tra.dat');data_tr = data_tr.data;
data_te = importdata('wankara-5-5tst.dat');data_te = data_te.data;
data_all = [data_tr;data_te];p = size(data_tr,2) - 1; % p represents the dimension of the inputs

%% train the model based on training samples
% parameter settings of DJFNN
order = 1; % order of consequnce in DJFNN
fn = 3*ones(1,p);% number of fuzzy sets along each dim
C = 9; % number of clusters

% uniform-frequency fuzzification along each dim based on all (traning and test) samples
fuzzi = fuzzification( data_tr(:,1:p),fn,1);

% train the model based on training sampels
x_tr = data_tr(:,1:p);y_tr = data_tr(:,end);
model = DJFNN ( x_tr, y_tr, fuzzi,C,order );        

%% predicting at test inputs and calculate the mean squared error
x_te = data_te(:,1:p);y_te = data_te(:,end);
[B,y_pre] = DJFNN_Pre ( model, x_te );
mse = MSE(y_te,y_pre);

figure;plot(y_te,y_pre,'k.');
xlabel('Actual test outputs');ylabel('Predicted outputs');

%% plot the rule base
S_norm = @( a, b ) b.*(a==0) + a.*(b==0) + 1.*(a~=0 & b~=0);
C = model.C;Fuzzi = model.Fuzzi;Q = model.Q;A = model.A;dim = model.dim;
scale = model.scale;intercept = model.intercept;order = model.Order;

M = zeros(1,dim);Par = cell(1,dim);lb = zeros(1,dim);ub = lb;
for j =  1:dim
    M(j) = Fuzzi.input{j}.MFsNum;
    Par{j} = Fuzzi.input{j}.MFsPar*scale(j) + intercept(j);
    lb(1,j) = Par{j}(1);ub(1,j) = Par{j}(end);
end

% Antecedents
figure
for i = 1 : C
    for j = 1 : dim
        subplot(C,dim,dim*(i-1) + j);
        xx = linspace(lb(j),ub(j),1000);
        yy0 = zeros(1,1000);
        for k = 1 : M(j)
            yy_temp = trapmf(xx,Par{j}(k,:));
            yy_temp = yy_temp*Q{j}(k,i);
            yy = S_norm(yy0,yy_temp);
            yy0 = yy;
        end
        plot(xx,yy);axis([lb(j),ub(j),0,1.2]);
    end
end
suptitle('Antecedents for each input variables (column) in each rule (row)');
% Consequents (absolute value which is normalized to [0,1])
ConPar = reshape(A,1+dim,C);
weights_norm =  abs(ConPar(2:end,:)) ./ max(abs(ConPar(2:end,:)));




