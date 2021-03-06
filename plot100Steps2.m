function plot100Steps2(vet_bin_acc, options)



window_length = round(length(vet_bin_acc)/100);
max = length(vet_bin_acc);

acc = [];
for i = 1 : window_length : max
    if i+window_length <= max
       acc =  [acc; sum(vet_bin_acc(i:i+window_length-1)/window_length)*100];
    else
       acc =  [acc; sum(vet_bin_acc(i:max)/length(vet_bin_acc(i:max)))*100];
    end
end

plot(acc, options, 'LineWidth',2); hold on;

set(gca, 'FontSize', 18);
xlabel('Passo', 'FontSize', 20);
ylabel('Acuracia (%)', 'FontSize', 20);
axis([0 100 0 100]);


% possible options:
% -ro
% -*k
% -ob