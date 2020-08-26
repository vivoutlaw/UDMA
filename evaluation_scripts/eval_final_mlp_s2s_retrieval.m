function eval_final_mlp_s2s_retrieval(dataset_name_, model_name_, epoch_number_, method_type_, eval_type_, L1_L12_L123_FinchPartition, MLP_epoch_number_)

% Consumer is query set
% Shop is gallery set


path_to_feats_ = '../features';
path_to_mlp_feats_ = sprintf('%s/MLP_G_Weighting/%s/%s_%s',path_to_feats_,dataset_name_,model_name_,L1_L12_L123_FinchPartition);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Query ------------------ Features
filename = sprintf('%s/%s/%s/Query_%s_%s_Crop.h5',path_to_feats_,dataset_name_,model_name_,method_type_,num2str(epoch_number_));
query_gt_class_labels = double(hdf5read(filename,'class_labels')) +1;
query_product_labels = double(hdf5read(filename,'labels'));

filename_feats = sprintf('%s/Query_%s_%s_Crop.h5',path_to_mlp_feats_,method_type_,num2str(MLP_epoch_number_));
query_all_feats = hdf5read(filename_feats,'X_').';
clear feats gt_class_labels labels


% Gallery ------------------ Features
filename = sprintf('%s/%s/%s/Gallery_%s_%s_Crop.h5',path_to_feats_,dataset_name_,model_name_,method_type_,num2str(epoch_number_));
gallery_gt_class_labels = double(hdf5read(filename,'class_labels')) +1;
gallery_product_labels = double(hdf5read(filename,'labels'));

filename_feats = sprintf('%s/Gallery_%s_%s_Crop.h5',path_to_mlp_feats_,method_type_,num2str(MLP_epoch_number_));
gallery_all_feats = hdf5read(filename_feats,'X_').';
clear feats gt_class_labels labels


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_class_categories = max(unique(gallery_gt_class_labels));
rank_upto = 6;
map_cmc_all = zeros([num_class_categories, rank_upto]);

for kk_ =1:num_class_categories
 
    % Labels and features: Query
    idx_class_query_k = find(query_gt_class_labels == kk_);
    query_product_labels_k = query_product_labels(idx_class_query_k);
    
    % Labels and features: Gallery
    idx_class_gallery_k = find(gallery_gt_class_labels == kk_);
    gallery_product_labels_k = gallery_product_labels(idx_class_gallery_k);
    
   
   % Removing samples of product ids in query set, that is not exising in gallery set 
   % This just happens for one class
   non_overlapping_product_ids = setdiff(query_product_labels_k,gallery_product_labels_k);
   if size(non_overlapping_product_ids,2) > 0
       idx_to_remove = find(query_product_labels_k == non_overlapping_product_ids);
       query_product_labels_k(idx_to_remove) = [];
       idx_class_query_k(idx_to_remove) = [];
   end
   
    
    % Features and product ids
    
    query_feats = query_all_feats(idx_class_query_k,:);
    queryID = query_product_labels_k;
    
    gallery_feats = gallery_all_feats(idx_class_gallery_k,:);
    testID = gallery_product_labels_k;    
    
    nQuery=size(queryID,2);
    
    if strcmp(eval_type_,'regular')
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
    %fprintf('Performance:  mAP = %0.4f, r1 precision = %0.4f,  r5 precision = %0.4f,  r10 precision = %0.4f, r20 precision = %0.4f, r50 precision = %0.4f\r\n', mAP, CMC(1), CMC(5), CMC(10), CMC(20), CMC(50));
    map_cmc_all(kk_,1:rank_upto) = [mAP, CMC(1), CMC(5), CMC(10), CMC(20), CMC(50)];
    clear ap score CMC d    

end

cmc_all = mean(map_cmc_all);
fprintf('All Performance:  mAP = %0.4f, r1 precision = %0.4f,  r5 precision = %0.4f,  r10 precision = %0.4f, r20 precision = %0.4f, r50 precision = %0.4f\r\n', cmc_all(1), cmc_all(2), cmc_all(3), cmc_all(4), cmc_all(5), cmc_all(6));

remove_classes_ = [1,2,4,5,6];
% remove_classes_ = [3,7,8,9,10,11];
map_cmc_all(remove_classes_,:) = [];
for ii_= 1:size(map_cmc_all,1)
    mAP =  map_cmc_all(ii_,1);
    CMC = map_cmc_all(ii_,2:end);
    fprintf('Performance:  mAP = %0.4f, r1 precision = %0.4f,  r5 precision = %0.4f,  r10 precision = %0.4f, r20 precision = %0.4f, r50 precision = %0.4f\r', mAP, CMC(1), CMC(2), CMC(3), CMC(4), CMC(5));
end
cmc_all = mean(map_cmc_all);
fprintf('Selected Performance:  mAP = %0.4f, r1 precision = %0.4f,  r5 precision = %0.4f,  r10 precision = %0.4f, r20 precision = %0.4f, r50 precision = %0.4f\r\n', cmc_all(1), cmc_all(2), cmc_all(3), cmc_all(4), cmc_all(5), cmc_all(6));
    
end









