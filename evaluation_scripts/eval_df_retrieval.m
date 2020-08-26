function eval_df_retrieval(dataset_name_, model_name_, epoch_number_, method_type_, eval_type)

path_to_feats_ = '../features';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Query ------------------ LABELS 
filename = sprintf('%s/%s/%s/Query_Labels_%s.txt',path_to_feats_,dataset_name_,model_name_,num2str(epoch_number_));
fileID = fopen(filename,'r');
dataArray = textscan(fileID,'%s','Delimiter',' ')';
query_categorical_labels = dataArray{1, 1};

query = categorical(query_categorical_labels);
query_idx = grp2idx(query);


% Gallery ------------------ LABELS 
filename = sprintf('%s/%s/%s/Gallery_Labels_%s.txt',path_to_feats_,dataset_name_,model_name_,num2str(epoch_number_));
fileID = fopen(filename,'r');
dataArray = textscan(fileID,'%s','Delimiter',' ')';
gallery_categorical_labels = dataArray{1, 1};

gallery = categorical(gallery_categorical_labels);
gallery_idx = grp2idx(gallery);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Query ------------------ Features 
filename = sprintf('%s/%s/%s/Query_%s_%s.mat',path_to_feats_,dataset_name_,model_name_,method_type_,num2str(epoch_number_));
load(filename)
query_feats = feats;

% % Gallery ------------------ Features 
filename = sprintf('%s/%s/%s/Gallery_%s_%s.mat',path_to_feats_,dataset_name_,model_name_,method_type_,num2str(epoch_number_));
load(filename)
gallery_feats = feats;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluation
queryID = query_idx;
testID = gallery_idx;

nQuery=size(queryID,1);
    
if strcmp(eval_type,'regular')
    d=pdist2(query_feats, gallery_feats,'cosine').';
end

% Compute Accuracy
%fprintf("Yes, we made it!\n")
junk_index=[];
parfor k=1:nQuery
    good_index = find(testID == queryID(k));
    score=d(:,k);
   % sort database images according distance
    [~, index] = sort(score, 'ascend');  % single query 
    [ap(k), CMC(k, :)] = compute_AP(good_index, junk_index, index);    
end

mAP=mean(ap);
CMC = mean(CMC);
fprintf('Performance:  mAP = %0.4f, r1 precision = %0.4f,  r5 precision = %0.4f,  r10 precision = %0.4f, r20 precision = %0.4f,  r50 precision = %0.4f\r\n', mAP, CMC(1), CMC(5), CMC(10), CMC(20),CMC(50));

    
end










