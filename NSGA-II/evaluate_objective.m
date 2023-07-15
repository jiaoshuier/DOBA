function f = evaluate_objective(individual,train_instances,valid_instances,pmax,nmax)

%% function f = evaluate_objective(x, M, V)
% Function to evaluate the objective functions for the given input vector
% x. x is an array of decision variables and f(1), f(2), etc are the
% objective functions. The algorithm always minimizes the objective
% function hence if you would like to maximize the function then multiply
% the function by negative one. M is the numebr of objective functions and
% V is the number of decision variables.
%
% This functions is basically written by the user who defines his/her own
% objective function. Make sure that the M and V matches your initial user
% input. Make sure that the
%
% An example objective function is given below. It has two six decision
% variables are two objective functions.

% f = [];
% %% Objective function one
% % Decision variables are used to form the objective function.
% f(1) = 1 - exp(-4*x(1))*(sin(6*pi*x(1)))^6;
% sum = 0;
% for i = 2 : 6
%     sum = sum + x(i)/4;
% end
% %% Intermediate function
% g_x = 1 + 9*(sum)^(0.25);
%
% %% Objective function two
% f(2) = g_x*(1 - ((f(1))/(g_x))^2);

%% Kursawe proposed by Frank Kursawe.
% Take a look at the following reference
% A variant of evolution strategies for vector optimization.
% In H. P. Schwefel and R. Männer, editors, Parallel Problem Solving from
% Nature. 1st Workshop, PPSN I, volume 496 of Lecture Notes in Computer
% Science, pages 193-197, Berlin, Germany, oct 1991. Springer-Verlag.
%
% Number of objective is two, while it can have arbirtarly many decision
% variables within the range -5 and 5. Common number of variables is 3.

% f = [];
% % Objective function one
% sum = 0;
% for i = 1 : V - 1
%     sum = sum - 10*exp(-0.2*sqrt((x(i))^2 + (x(i + 1))^2));
% end
% % Decision variables are used to form the objective function.
% f(1) = sum;
%
% % Objective function two
% sum = 0;
% for i = 1 : V
%     sum = sum + (abs(x(i))^0.8 + 5*(sin(x(i)))^3);
% end
% % Decision variables are used to form the objective function.
% f(2) = sum;
%
%
%
% %% Check for error
% if length(f) ~= M
%     error('The number of decision variables does not match you previous input. Kindly check your objective function');
% end

%%
a = individual(1);
b = individual(2);
valid_data = valid_instances(:,1:end-1);
valid_labels = valid_instances(:,end);
h1_test = sum(valid_labels==1);
h0_test = sum(valid_labels==0);
[score_valid] = ComputeTWSVMScore(train_instances,valid_data);
pred_valid = zeros(size(valid_labels))+2;
            pred_valid(score_valid>a,:) = 1;
            pred_valid(score_valid<b,:) = 0;
            rpr = sum(pred_valid==2 & valid_labels==1)/h1_test;
            rnr = sum(pred_valid==2 & valid_labels==0)/h0_test;
    if rpr<=pmax && rnr<=nmax 
        tp = sum(pred_valid==1 & valid_labels==1);
        fn = sum(pred_valid==0 & valid_labels==1);
        tn = sum(pred_valid==0 & valid_labels==0);
        fp = sum(pred_valid==1 & valid_labels==0);
        TPR = tp/(tp+fn);
        TNR = tn/(tn+fp);
        if isnan(TPR) || isnan(TNR)
            TPR = 0;
            TNR = 0;
        end
    else
        TPR = 0;
        TNR = 0;
    end
f = [1-mean(TPR) 1-mean(TNR)];    
%%    

