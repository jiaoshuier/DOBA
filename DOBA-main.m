
pop = 40;
gen = 100;


        load(file);   % file contains train_instances, valid_instances, and test_instances
        train_data = train_instances(:,1:end-1);
        [score_train] = ComputeTWSVMScore(train_instances,train_data);
        minv = min(score_train);
        maxv = max(score_train);
        
        test_data = test_instances(:,1:end-1); 
        [score_test] = ComputeTWSVMScore(train_instances,test_data);
        test_labels = test_instances(:,end);   % 1 is positive class and 0 is negative class
        h1_test = sum(test_labels==1);
        h0_test = sum(test_labels==0);
        
        val_data = valid_instances(:,1:end-1); 
        [score_val] = ComputeTWSVMScore(train_instances,val_data);
        val_labels = valid_instances(:,end);
        
        %% multi-objective optimization
        tic;
         kmax = 0.01:0.02:0.99;     %%%%%%%%%%%%%%%%%%%%%%%%    modify para
        population = nsga_2(pop,gen,train_instances,valid_instances,minv,maxv);
        toc;
        for tt = 1:50  % para = 0.01:0.02:0.99   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  modify para
            % using val set to find the best abstaining classifier (decision variables)
            for j = 1:pop
                a = population{tt}(j,1);
                b = population{tt}(j,2);
                pred_val = zeros(size(val_labels))+2;
                pred_val(score_val>a,:) = 1;
                pred_val(score_val<b,:) = 0;
                rej(j) = sum(pred_val==2)/length(val_labels);
                rpr(j) = sum(pred_val==2 & val_labels==1)/sum(val_labels==1);
                rnr(j) = sum(pred_val==2 & val_labels==0)/sum(val_labels==0);
                tp = sum(pred_val==1 & val_labels==1);
                fn = sum(pred_val==0 & val_labels==1);
                tn = sum(pred_val==0 & val_labels==0);
                fp = sum(pred_val==1 & val_labels==0);
                acc(j) = (tp+tn)/(fp+tn+fn+tp);
                sen(j) = tp/(tp+fn);
                spe(j) = tn/(tn+fp);
                auc(j) = (sen(j)+spe(j))/2;
                g(j) = sqrt(sen(j)*spe(j));
            end
            id = find( rpr<= kmax(tt) & rnr <=kmax(tt));
            if isempty(id)
                R{tt,runs} = [];
                continue;
            else
                [~,in1]= max(acc(id));
                [~,in2] = max(auc(id));
                [~,in3] = max(g(id));
                para = [population{tt}(id(in1),1:2);population{tt}(id(in2),1:2);population{tt}(id(in3),1:2)];
            end
            temp = [];
            for index = 1:3
                pred_test = zeros(size(test_labels))+2;
                pred_test(score_test>para(index,1),:) = 1;
                pred_test(score_test<para(index,2),:) = 0;
                Rej = sum(pred_test==2)/length(test_labels);
                RPR = sum(pred_test==2 & test_labels==1)/sum(test_labels==1);
                RNR = sum(pred_test==2 & test_labels==0)/sum(test_labels==0);
                tp = sum(pred_test==1 & test_labels==1);
                fn = sum(pred_test==0 & test_labels==1);
                tn = sum(pred_test==0 & test_labels==0);
                fp = sum(pred_test==1 & test_labels==0);
                ACC = (tp+tn)/(fp+tn+fn+tp);
                SEN = tp/(tp+fn);
                SPE = tn/(tn+fp);
                AUC = (SEN+SPE)/2;
                G = sqrt(SEN*SPE);
                temp = [temp; ACC AUC G SEN SPE Rej RPR RNR];
            end
            R{tt,runs} = temp;  
        end
               

