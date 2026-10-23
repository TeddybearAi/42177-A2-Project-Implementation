function result = linearProbe(cfg)
%LINEARPROBE Exploratory run R2p: linear head on cached, unaugmented frozen ViT features.
%   Extracts the class-token features (input of the "head" layer) of the pretrained
%   ViT once per image with the deterministic evaluation transform, trains a 12-way
%   linear layer on the training features (Adam 1e-3, mini-batch 128, 30 epochs,
%   seed 42, lowest validation loss kept), then evaluates once on E1 and E2.
%   See the R2p amendment in docs/adr/0002. Writes outputs/runs/R2p/ in the same
%   layout as trainRun (results.json, predictions, per-class tables, confusion CSVs).
arguments
    cfg = vitConfig()
end
seed = 42;
outDir = fullfile(cfg.outDir, "runs", "R2p");
if ~isfolder(outDir), mkdir(outDir); end
rng(seed, "twister");
gpurng(seed);

[M, classNames] = loadManifest(cfg);
nClasses = numel(classNames);
net = visionTransformer(cfg.model);
featureLayer = net.Layers(find(string({net.Layers.Name}) == "head") - 1).Name;   % layer feeding the classifier

% ---- frozen features, extracted once per partition (no augmentation anywhere)
tFeat = tic;
F = struct();
for s = ["train", "val", "test", "external"]
    rows = M(M.split == s, :);
    ds = imageDatastore(rows.Path, "ReadFcn", @(f) preprocessImage(f, "eval", Config=cfg));
    F.(s) = gather(minibatchpredict(net, ds, "MiniBatchSize", 32, "Outputs", featureLayer));
    assert(size(F.(s), 1) == height(rows), "feature rows do not match partition %s", s);
    fprintf("%s: %d x %d features\n", s, size(F.(s), 1), size(F.(s), 2));
end
featureMinutes = toc(tFeat) / 60;

% ---- linear head on cached features
tr = M(M.split == "train", :); va = M(M.split == "val", :);
head = dlnetwork([featureInputLayer(size(F.train, 2)), fullyConnectedLayer(nClasses), softmaxLayer]);
options = trainingOptions("adam", ...
    "InitialLearnRate", 1e-3, ...
    "MiniBatchSize", 128, ...
    "MaxEpochs", 30, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", {F.val, va.Y}, ...
    "ValidationFrequency", ceil(height(tr) / 128), ...   % once per epoch
    "OutputNetwork", "best-validation", ...
    "Plots", "none", ...
    "Verbose", true, ...
    "VerboseFrequency", ceil(height(tr) / 128));
tStart = tic;
[head, info] = trainnet(F.train, tr.Y, head, "crossentropy", options);
trainMinutes = toc(tStart) / 60;
writetable(info.TrainingHistory, fullfile(outDir, "training_history.csv"));
writetable(info.ValidationHistory, fullfile(outDir, "validation_history.csv"));

% ---- val (for reference) and the locked E1 / E2 evaluations
run = struct("id", "R2p", "regime", "linear probe", "grayWorld", false, "seed", seed);
[~, pVal] = max(predict(head, F.val), [], 2);
mVal = classificationMetrics(double(va.Y), pVal, nClasses);
e1 = evaluateFeatures(head, F.test, M(M.split == "test", :), classNames, "internal", outDir, seed, cfg);
e2 = evaluateFeatures(head, F.external, M(M.split == "external", :), classNames, "external", outDir, seed, cfg);

result = struct("run", run, "model", cfg.model, "featureLayer", featureLayer, ...
    "learnRate", 1e-3, "miniBatchSize", 128, "maxEpochs", 30, ...
    "selectedValF1Macro", mVal.f1Macro, "selectedValAccuracy", mVal.accuracy, ...
    "E1", e1, "E2", e2, "gapF1Macro", e1.f1Macro - e2.f1Macro, ...
    "retentionF1Macro", e2.f1Macro / e1.f1Macro, ...
    "featureMinutes", featureMinutes, "trainMinutes", trainMinutes, ...
    "finished", string(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss Z", "TimeZone", "local")));
fid = fopen(fullfile(outDir, "results.json"), "w");
fprintf(fid, "%s\n", jsonencode(result, "PrettyPrint", true));
fclose(fid);
fprintf("R2p done: val macro-F1 %.4f | E1 macro-F1 %.4f | E2 macro-F1 %.4f [%.3f, %.3f]\n", ...
    mVal.f1Macro, e1.f1Macro, e2.f1Macro, e2.f1MacroCI95);
end

function summary = evaluateFeatures(head, X, rows, classNames, name, outDir, seed, cfg)
% Same outputs as evaluateSplit (prediction file in the group format, per-class and
% confusion CSVs, bootstrap CI), from cached features instead of image files.
nClasses = numel(classNames);
scores = predict(head, X);
[~, yPred] = max(scores, [], 2);
yTrue = double(rows.Y);
m = classificationMetrics(yTrue, yPred, nClasses);
P = table(rows.image_id, yTrue - 1, yPred - 1, 'VariableNames', ["image_id", "y_true", "y_pred"]);
P = [P, array2table(scores, 'VariableNames', "p_" + string(0:nClasses - 1))];
assert(height(P) == numel(unique(P.image_id)) && all(P.y_true == rows.y), "prediction rows do not match manifest");
writetable(P, fullfile(outDir, "predictions_" + name + ".csv"));
perClass = m.perClass;
perClass.Properties.RowNames = classNames;
writetable(perClass, fullfile(outDir, "per_class_" + name + ".csv"), 'WriteRowNames', true);
writetable(array2table(m.confusion, 'VariableNames', classNames, 'RowNames', classNames), ...
    fullfile(outDir, "confusion_" + name + ".csv"), 'WriteRowNames', true);
summary = rmfield(m, ["perClass", "confusion"]);
summary.f1MacroCI95 = bootstrapF1(yTrue, yPred, nClasses, seed, cfg.nBoot);
end
