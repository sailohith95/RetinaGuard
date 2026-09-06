%% evaluate_classifier.m
%
%   Evaluation script for the DR severity classifier.
%   Reports accuracy, sensitivity, specificity, F1, and confusion matrix.
%
%   PREREQUISITES:
%   - Trained model at models/dr/dr_classifier.mat
%   - APTOS 2019 test set (held-out from training)
%
%   IMPORTANT: Do NOT use training labels for evaluation.
%   Evaluation is only meaningful on UNSEEN test data.

fprintf('RetinaGuard — DR Classifier Evaluation\n');
fprintf('========================================\n\n');

cfg = RGConfig();

% ---- Check model ----------------------------------------------------
if ~isfile(cfg.paths.drModel)
    error(['Trained model not found at: %s\n' ...
           'Train the model first using training/train_dr_classifier.m'], cfg.paths.drModel);
end

data = load(cfg.paths.drModel, 'net', 'classes');
net  = data.net;

fprintf('Model loaded: %s\n', cfg.paths.drModel);

% ---- Load test set --------------------------------------------------
% NOTE: The test set must be completely separate from training data.
% Use the IDRiD test set or a held-out portion of APTOS.

testCSV = fullfile(cfg.paths.aptos, 'test.csv');
testDir = fullfile(cfg.paths.aptos, 'test_images');

if ~isfile(testCSV) || ~isfolder(testDir)
    error('Test set not found. See DATASET_SETUP.md and EVALUATION.md.');
end

T      = readtable(testCSV);
imgIDs = T.id_code;
gtLabels = T.diagnosis;

% ---- Run inference --------------------------------------------------
fprintf('Running inference on %d test images...\n', height(T));
tSize  = cfg.preprocess.targetSize;
predLabels = zeros(height(T), 1);
predConf   = zeros(height(T), 1);

for k = 1:height(T)
    imgPath = fullfile(testDir, [imgIDs{k} '.png']);
    if ~isfile(imgPath)
        imgPath = strrep(imgPath, '.png', '.jpg');
    end
    if ~isfile(imgPath)
        predLabels(k) = 0; predConf(k) = 0; continue;
    end
    img = imresize(imread(imgPath), tSize);
    if size(img,3)==1, img = cat(3,img,img,img); end
    scores = predict(net, single(img));
    [conf, idx] = max(scores);
    predLabels(k) = idx - 1;  % 0-indexed
    predConf(k)   = conf;
end

% ---- Metrics --------------------------------------------------------
gtCat   = categorical(gtLabels, 0:4);
predCat = categorical(predLabels, 0:4);

fprintf('\n--- Overall Metrics ---\n');
acc = sum(predLabels == gtLabels) / numel(gtLabels);
fprintf('Accuracy:    %.2f%%\n', acc*100);

cm = confusionmat(gtLabels, predLabels);
fprintf('\nConfusion Matrix (rows=actual, cols=predicted, grades 0-4):\n');
disp(cm);

% Per-class sensitivity and specificity
fprintf('\n--- Per-Class Metrics ---\n');
fprintf('%-12s  %s    %s    %s    %s\n', 'Grade', 'Sensitivity', 'Specificity', 'F1', 'Support');
for g = 0:4
    TP = cm(g+1,g+1);
    FN = sum(cm(g+1,:)) - TP;
    FP = sum(cm(:,g+1)) - TP;
    TN = sum(cm(:)) - TP - FN - FP;
    sens = TP / (TP + FN + eps);
    spec = TN / (TN + FP + eps);
    prec = TP / (TP + FP + eps);
    f1   = 2*prec*sens / (prec + sens + eps);
    fprintf('Grade %-2d:    Sens=%.2f%%   Spec=%.2f%%   F1=%.2f   N=%d\n', ...
        g, sens*100, spec*100, f1, sum(cm(g+1,:)));
end

% Referral performance (any DR vs no DR)
fprintf('\n--- Referral Threshold Performance (Grade >= 1 = Refer) ---\n');
anyDR_gt   = gtLabels >= 1;
anyDR_pred = predLabels >= 1;
TP_r = sum(anyDR_gt & anyDR_pred);
FN_r = sum(anyDR_gt & ~anyDR_pred);
FP_r = sum(~anyDR_gt & anyDR_pred);
TN_r = sum(~anyDR_gt & ~anyDR_pred);
fprintf('Sensitivity (DR detected): %.2f%%\n', 100*TP_r/(TP_r+FN_r+eps));
fprintf('Specificity (no-DR correct): %.2f%%\n', 100*TN_r/(TN_r+FP_r+eps));

fprintf('\nNOTE: These metrics are only valid on a properly held-out test set.\n');
fprintf('Do not report training-set or validation-set metrics as test performance.\n');
