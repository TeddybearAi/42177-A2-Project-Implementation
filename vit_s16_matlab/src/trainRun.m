function result = trainRun(runId, opts)
%TRAINRUN Train, select and evaluate one pre-specified run (R1..R5, docs/adr/0002).
%   result = trainRun("R1")                     full run
%   result = trainRun("R1", Smoke=true)         few images, 2 epochs: checks the pipeline
%   result = trainRun("R1", MiniBatchSize=6)    protocol fallback when GPU memory runs out
%   result = trainRun("R1", Model="tiny-16-imagenet-384")   second fallback (all runs)
%
%   1. Seeds MATLAB and GPU random streams with the run's seed.
%   2. Loads ViT, replaces "head" with a 12-class layer, freezes per regime
%      (frozen: head only; attention: self-attention layers + head) via learn-rate factor 0.
%   3. trainnet with the MathWorks recipe (Adam 1e-4, 4 epochs), saving a checkpoint each epoch.
%   4. Checkpoint selection: macro-F1 of every epoch's checkpoint on PlantVillage val;
%      best kept (earliest on ties). Only then are test (E1) and external (E2) touched.
%   Outputs: outputs/runs/<id>/ (results.json, selection.csv, training_history.csv,
%   predictions, per-class tables, confusion matrices); weights in outputs/checkpoints/<id>/.
arguments
    runId (1,1) string
    opts.Smoke (1,1) logical = false
    opts.MiniBatchSize (1,1) double = NaN
    opts.Model (1,1) string = ""
end
cfg = vitConfig();
if ~isnan(opts.MiniBatchSize), cfg.miniBatchSize = opts.MiniBatchSize; end
if opts.Model ~= "", cfg.model = opts.Model; end
run = cfg.runs([cfg.runs.id] == runId);
assert(isscalar(run), "Unknown run %s", runId);
if ~isnan(run.learnRate), cfg.learnRate = run.learnRate; end
name = run.id;
if opts.Smoke, name = "smoke_" + run.id; end
outDir = fullfile(cfg.outDir, "runs", name);
ckptDir = fullfile(cfg.outDir, "checkpoints", name);
if isfolder(ckptDir), rmdir(ckptDir, "s"); end
mkdir(ckptDir);
if ~isfolder(outDir), mkdir(outDir); end

rng(run.seed, "twister");
gpurng(run.seed);

% ---- data: partitions straight from the shared manifest
[M, classNames] = loadManifest(cfg);
nClasses = numel(classNames);
part = @(s) M(M.split == s, :);
tr = part("train"); va = part("val"); te = part("test"); ex = part("external");
maxEpochs = cfg.maxEpochs;
if opts.Smoke   % pipeline check only: small random subsets, 2 epochs
    pick = @(T, k) T(sort(randperm(height(T), k)), :);
    tr = pick(tr, 120); va = pick(va, 48); te = pick(te, 48); ex = pick(ex, 48);
    maxEpochs = 2;
end
imdsTrain = imageDatastore(tr.Path, "Labels", tr.Y, ...
    "ReadFcn", @(f) preprocessImage(f, "train", GrayWorld=run.grayWorld, Config=cfg));

% ---- network: new 12-class head, regime-specific freezing
net = visionTransformer(cfg.model);
net = replaceLayer(net, "head", fullyConnectedLayer(nClasses, "Name", "head"));
net = initialize(net);                       % new head drawn from the seeded stream
layerNames = string({net.Layers.Name});
isAttention = arrayfun(@(l) isa(l, "nnet.cnn.layer.SelfAttentionLayer"), net.Layers)';
trainable = "head";
if run.regime == "attention", trainable = [trainable, layerNames(isAttention)]; end
L = net.Learnables;
nTrainable = 0;
for i = 1:height(L)
    if ismember(L.Layer(i), trainable)
        nTrainable = nTrainable + numel(L.Value{i});
    else
        net = setLearnRateFactor(net, L.Layer(i), L.Parameter(i), 0);
    end
end
fprintf("%s: %s, regime %s, gray-world %d, seed %d, batch %d, learning rate %g, %d epochs; trainable %.2fM of %.2fM parameters\n", ...
    name, cfg.model, run.regime, run.grayWorld, run.seed, cfg.miniBatchSize, cfg.learnRate, maxEpochs, ...
    nTrainable / 1e6, sum(cellfun(@numel, L.Value)) / 1e6);

% ---- training (MathWorks ViT recipe), one checkpoint per epoch
options = trainingOptions("adam", ...
    "InitialLearnRate", cfg.learnRate, ...
    "MiniBatchSize", cfg.miniBatchSize, ...
    "MaxEpochs", maxEpochs, ...
    "Shuffle", "every-epoch", ...
    "CheckpointPath", ckptDir, ...
    "CheckpointFrequency", 1, ...
    "CheckpointFrequencyUnit", "epoch", ...
    "OutputNetwork", "last-iteration", ...
    "PreprocessingEnvironment", "parallel", ...   % augmentation (~50 ms/image) on a local pool of CPU workers
    "Plots", "none", ...
    "Verbose", true, ...
    "VerboseFrequency", 100);
if isempty(gcp("nocreate")), parpool("Processes", cfg.preprocessWorkers); end
tStart = tic;
[~, info] = trainnet(imdsTrain, net, "crossentropy", options);
trainMinutes = toc(tStart) / 60;
writetable(info.TrainingHistory, fullfile(outDir, "training_history.csv"));

% ---- checkpoint selection on PlantVillage val macro-F1 only
files = dir(fullfile(ckptDir, "*.mat"));
iteration = cellfun(@(n) str2double(regexp(n, "__(\d+)__", "tokens", "once")), {files.name})';
[iteration, order] = sort(iteration);
files = files(order);
epoch = (1:numel(files))';
valF1 = zeros(numel(files), 1);
valAccuracy = zeros(numel(files), 1);
for k = 1:numel(files)
    ckNet = loadNetwork(fullfile(ckptDir, files(k).name));
    [~, p] = max(predictScores(ckNet, va.Path, run.grayWorld, cfg), [], 2);
    m = classificationMetrics(double(va.Y), p, nClasses);
    valF1(k) = m.f1Macro;
    valAccuracy(k) = m.accuracy;
    fprintf("  epoch %d (iteration %d): val macro-F1 %.4f, accuracy %.4f\n", epoch(k), iteration(k), valF1(k), valAccuracy(k));
end
selection = table(epoch, iteration, valF1, valAccuracy, string({files.name})', ...
    'VariableNames', ["epoch", "iteration", "val_f1_macro", "val_accuracy", "checkpoint"]);
writetable(selection, fullfile(outDir, "selection.csv"));
[~, best] = max(valF1);                      % max returns the first (earliest) maximum
net = loadNetwork(fullfile(ckptDir, files(best).name));
save(fullfile(ckptDir, "selected.mat"), "net", "classNames");

% ---- locked evaluation: E1 (PlantVillage test) and E2 (Cropped-PlantDoc external), once
tEval = tic;
e1 = evaluateSplit(net, te, classNames, "internal", "PlantVillage internal test (E1)", outDir, run, cfg);
e2 = evaluateSplit(net, ex, classNames, "external", "Cropped-PlantDoc external test (E2)", outDir, run, cfg);

g = gpuDevice;
result = struct("run", run, "smoke", opts.Smoke, "model", cfg.model, "learnRate", cfg.learnRate, ...
    "miniBatchSize", cfg.miniBatchSize, "maxEpochs", maxEpochs, "trainableParamsMillion", nTrainable / 1e6, ...
    "nImages", struct("train", height(tr), "val", height(va), "test", height(te), "external", height(ex)), ...
    "selectedEpoch", epoch(best), "selectedValF1Macro", valF1(best), ...
    "E1", e1, "E2", e2, "gapF1Macro", e1.f1Macro - e2.f1Macro, ...
    "retentionF1Macro", ratioOrNaN(e2.f1Macro, e1.f1Macro), ...
    "trainMinutes", trainMinutes, "evalMinutes", toc(tEval) / 60, ...
    "gpu", string(g.Name), "matlab", string(version), ...
    "finished", string(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss Z", "TimeZone", "local")));
fid = fopen(fullfile(outDir, "results.json"), "w");
fprintf(fid, "%s\n", jsonencode(result, "PrettyPrint", true));
fclose(fid);
fprintf("%s done: selected epoch %d (val F1 %.4f) | E1 macro-F1 %.4f | E2 macro-F1 %.4f [%.3f, %.3f] | %.1f min training\n", ...
    name, epoch(best), valF1(best), e1.f1Macro, e2.f1Macro, e2.f1MacroCI95, trainMinutes);
end

function net = loadNetwork(file)
% Checkpoint files hold one dlnetwork variable; take it whatever it is called.
S = load(file);
f = fieldnames(S);
net = S.(f{find(cellfun(@(n) isa(S.(n), "dlnetwork"), f), 1)});
end

function r = ratioOrNaN(a, b)
if b > 0, r = a / b; else, r = NaN; end
end
