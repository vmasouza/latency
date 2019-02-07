
function [vet_bin_acc, acc_final] = MClassification(dataset, numini, radiusThreshold) 


    data = load(dataset);
   
    %initial labeled data for create the initial Micro Clusters
    initial_labeled_DATA = data(1:numini,1:end-1);
    initial_labeled_LABELS = data(1:numini,end);    
  
    %Threshold de raio ??? o  mesmo para todas as dimens???es dos dados
    radiusThreshold = ones(1, size(initial_labeled_DATA,2)) * radiusThreshold;
    
    %unlabeled data used for the test phase
    unlabeled_DATA = data(numini+1:end, 1:end-1);
    unlabeled_LABELS = data(numini+1:end,end);
    
    classes = unique(initial_labeled_LABELS);
    classes = [classes, zeros(length(classes),1)];
    nClass = size(classes,1);
    
    nMC = 0;  
    for nC = 1 : numini
        nMC = nMC + 1;
        example = initial_labeled_DATA(nC, :);
        MC(nMC).id = nMC;
        MC(nMC).LS = example;
        MC(nMC).SS = example.^2;
        MC(nMC).N = 1;
        MC(nMC).centroid = calculateCentroid(MC(nMC).LS, MC(nMC).N);
        MC(nMC).radius = calculateRadius(MC(nMC).LS, MC(nMC).SS, MC(nMC).N);        
        MC(nMC).class = initial_labeled_LABELS(nC);
    end
        
    vet_bin_acc = [];
    for i = 1:length(unlabeled_LABELS)     
       test_instance = unlabeled_DATA(i,:);
       actual_label = unlabeled_LABELS(i);
       
       cents = [];
       for qwe = 1:length(MC)
           cents = [cents; MC(qwe).centroid, MC(qwe).id];
       end
       [positionMC,~]=knnsearch(cents(:,1:end-1), test_instance,'k',1);
       positionMC = find([MC.id] == cents(positionMC,end));
       nearestMC = MC(positionMC);
       predicted_label = MC(positionMC).class;

       
       %virtually adds an instance in the nearest MC (temporally)
       tempMC.id = nearestMC.id;
       tempMC.LS = nearestMC.LS + test_instance;
       tempMC.SS = nearestMC.SS + test_instance.^2;
       tempMC.N = nearestMC.N + 1;
       tempMC.centroid = calculateCentroid(tempMC.LS, tempMC.N);
       tempMC.radius = calculateRadius(tempMC.LS, tempMC.SS, tempMC.N);
       tempMC.class = nearestMC.class;
       
       if any(tempMC.radius > radiusThreshold)
                      
           %cria um novo MC
           nMC = nMC + 1;
           
           idxMC = [MC.class] == predicted_label;
           neighborhoodMCS = MC(idxMC);
           neighborhood  = [];
           for qwe = 1 :length(neighborhoodMCS)
               neighborhood = [neighborhood; neighborhoodMCS(qwe).centroid];
           end                    
            
           %if ~isempty(neighborhood)
           if length(neighborhood) > 20
% % %                [idxFarthest, ~] = knnsearch(neighborhood, test_instance, 'k', length(neighborhood));
% % %                idxFarthest = idxFarthest(end);   
% % %                farthestMC = neighborhoodMCS(idxFarthest);
% % %                idxFarthest = find([MC.id] == farthestMC.id);
% % %                MC(idxFarthest) = [];


               [idxFarthest, ~] = knnsearch(neighborhood, test_instance, 'k', length(neighborhood));
               
               idxFarthest2 = idxFarthest(end-1);
               idxFarthest = idxFarthest(end);

               
               farthestMC = neighborhoodMCS(idxFarthest);
               idxFarthest = find([MC.id] == farthestMC.id);

               farthestMC2 = neighborhoodMCS(idxFarthest2);
               idxFarthest2 = find([MC.id] == farthestMC2.id);
               
               %cria um novo MC (merge dos 2 mais distantes)
               pos = idxFarthest;
               MC(pos).id = MC(idxFarthest).id;
               MC(pos).LS = MC(idxFarthest).LS + MC(idxFarthest2).LS;
               MC(pos).SS = MC(idxFarthest).SS + MC(idxFarthest2).SS;
               MC(pos).N = MC(idxFarthest).N + MC(idxFarthest2).N;
               MC(pos).centroid = calculateCentroid(MC(pos).LS, MC(pos).N);
               MC(pos).radius = calculateRadius(MC(pos).LS, MC(pos).SS, MC(pos).N);
               MC(pos).class = predicted_label; 

               MC(idxFarthest2) = [];
             
           end

           pos = size(MC,2)+1;
           MC(pos).id = nMC;
           MC(pos).LS = test_instance;
           MC(pos).SS = test_instance.^2;
           MC(pos).N = 1;
           MC(pos).centroid = calculateCentroid(MC(pos).LS, MC(pos).N);
           MC(pos).radius = calculateRadius(MC(pos).LS, MC(pos).SS, MC(pos).N);
           MC(pos).class = predicted_label; 
       else
  
           %adiciona exemplo no MC mais proximo
           MC(positionMC) = tempMC;
       end       
       
       
       
% % %        cents = [];
% % %        for qwe = 1:length(MC)
% % %            cents = [cents; MC(qwe).centroid, MC(qwe).class];
% % %        end
% % %        class1 = find(cents(:,end) == 1);
% % %        class2 = find(cents(:,end) == 2);
% % %        class3 = find(cents(:,end) == 3);
% % %        class4 = find(cents(:,end) == 4);
% % %        plot(cents(class1,1), cents(class1,2), 'ro', cents(class2,1), cents(class2,2), 'bo');
% % %        plot(cents(class1,1), cents(class1,2), 'ro', cents(class2,1), cents(class2,2), 'bo', cents(class3,1), cents(class3,2), 'ko', cents(class4,1), cents(class4,2), 'mo');
% % %        axis([-10 10 -10 10]);
% % %        drawnow
       

       
       %update vet_bin_acc for calculate the accuracy measure
       if predicted_label == actual_label
            vet_bin_acc = [vet_bin_acc, 1];
       else
            vet_bin_acc = [vet_bin_acc, 0];
       end
    end
    
    acc_final = (sum(vet_bin_acc)/length(unlabeled_DATA))*100;
end


   

function raio = calculateRadius(LS, SS, N) %or standard deviation
    raio = sqrt( (SS./N) - (LS./N).^2 );
end

function centroide = calculateCentroid(LS, N) %or mean
    centroide = LS./N;
end

% % % function diametro = calculateDiameter(LS, SS, N)
% % %     diametro = sqrt((2*N*SS-2*(LS).^2)./N*(N-1));
% % % end
