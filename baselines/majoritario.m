function [vet_bin_acc, acc_final] = majoritario(dataset, MAX_TAM)


    
    data = load(dataset);

   
    train_labels = data(1:MAX_TAM,end);
    test_labels = data(MAX_TAM+1:end,end);
    
    %train_data = data(1:MAX_TAM, 1:end-1);
    %test_data = data(MAX_TAM+1:end, 1:end-1);
    
    
    vet_bin_acc = [];

    for i = 1 : length(test_labels)
     
        correct = test_labels(i);

        %verifica classe majoritaria       
        infos = tabulate(train_labels);
        [~,idxMaj] = max(infos(:,2));
        predicted_label = infos(idxMaj,1);
        
        %[predicted_label, ~, ~] = knn_classify(train_data, train_labels, test_instance);
        
        %train_data = [train_data; test_instance];
        train_labels = [train_labels; correct];
        
            
        
        if predicted_label == correct;
            vet_bin_acc = [vet_bin_acc, 1];
        else
            vet_bin_acc = [vet_bin_acc, 0];
        end

    end
    acc_final = (sum(vet_bin_acc)/length(vet_bin_acc))*100;        
