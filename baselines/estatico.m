function [vet_bin_acc, acc_final] = estatico(dataset, MAX_TAM)


    
    data = load(dataset);

    %MAX_TAM = round(size(data,1)*percent); %initial max tam of training data
   
    train_labels = data(1:MAX_TAM,end);
    test_labels = data(MAX_TAM+1:end,end);
    
    train_data = data(1:MAX_TAM, 1:end-1);
    test_data = data(MAX_TAM+1:end, 1:end-1);
    
    %acc = 0;
    %err = 0;
    vet_bin_acc = [];
    %vet_acc = [];
    for i = 1 : length(test_data)
        test_instance = test_data(i:i, :);
        correct = test_labels(i);

        %[predicted_label, ~, ~] = knn_classify(train_data, train_labels, test_instance);
        idx = knnsearch(train_data, test_instance,'k',1);
        predicted_label = train_labels(idx);
        
        if predicted_label == correct;
            %acc = acc+1;
            vet_bin_acc = [vet_bin_acc, 1];
            %display([num2str(acc+err), '- Acertou.  Acuracia parcial = ', num2str(acc/(acc+err))]);
        else
            %err = err+1;
            vet_bin_acc = [vet_bin_acc, 0];
            %display([num2str(acc+err), '- Errou.  Acuracia parcial = ', num2str(acc/(acc+err))]);
        end
%         if i == 1
%             vet_acc = [(acc/(acc+err))*100];
%         else
%             vet_acc = [vet_acc; (acc/(acc+err))*100];
%         end

    end
    acc_final = (sum(vet_bin_acc)/length(vet_bin_acc))*100;
%     acc = (acc/(acc+err))*100;
%     display(['Acuracia media = ', num2str(acc)]);
%     
%     local = pwd;
%     save([local, '/1NN_static_', dataset, '_' num2str(percent*100), '.mat']);

        
