function T = summarizeRuns(cfg)
%SUMMARIZERUNS Collect finished runs into the tables and figures used in the report.
%   Reads outputs/runs/<id>/results.json for every run that has one (unfinished runs are
%   left out, never estimated) and writes to outputs/summary/:
%     vit_runs.csv           one row per run
%     comparison.csv / .md   ViT next to the group's CNN summary (shared/results/model_summary.csv)
%     per_class_f1_E2.png    E2 per-class F1: ViT R1/R2 vs ResNet50 full/frozen
%     training_loss.png      training loss per run (moving average over 100 iterations)
%     examples_R1.png        the group's fixed external sample (shared/gradcam_sample.csv)
%                            with R1's prediction for each image
%     confusion_<id>_<split>.png   row-normalised confusion matrices for E1 and E2
%   The three attention-regime seeds (R1, R4, R5) are summarised as mean ± SD, as in
%   the CNN table; single-seed rows show their value with SD left empty.
arguments
    cfg = vitConfig()
end
outDir = fullfile(cfg.outDir, "summary");
if ~isfolder(outDir), mkdir(outDir); end
runDir = @(id) fullfile(cfg.outDir, "runs", id);
[~, classNames] = loadManifest(cfg);

% ---- one row per finished run (R2p comes from linearProbe, not trainRun; ADR 0002 amendment)
rows = {};
R = struct();
probe = struct("id", "R2p", "regime", "linear probe", "grayWorld", false, "seed", 42, "learnRate", 1e-3);
for run = [cfg.runs, probe]
    file = fullfile(runDir(run.id), "results.json");
    if ~isfile(file), continue; end
    r = jsondecode(fileread(file));
    if ~isfield(r, "selectedEpoch"), r.selectedEpoch = NaN; end   % R2p keeps the best-validation-loss epoch
    R.(run.id) = r;
    rows(end + 1, :) = {run.id, run.regime, run.grayWorld, run.seed, r.selectedEpoch, ...
        r.selectedValF1Macro, r.trainMinutes, r.E1.accuracy, r.E1.f1Macro, r.E2.accuracy, ...
        r.E2.precisionMacro, r.E2.recallMacro, r.E2.f1Macro, r.E2.f1MacroCI95(1), ...
        r.E2.f1MacroCI95(2), r.gapF1Macro, r.retentionF1Macro}; %#ok<AGROW>
end
assert(~isempty(rows), "No finished run under %s", fullfile(cfg.outDir, "runs"));
T = cell2table(rows, 'VariableNames', ["run", "regime", "gray_world", "seed", "selected_epoch", ...
    "val_f1_macro", "train_minutes", "E1_accuracy", "E1_f1_macro", "E2_accuracy", ...
    "E2_precision_macro", "E2_recall_macro", "E2_f1_macro", "E2_f1_ci_low", "E2_f1_ci_high", ...
    "gap_f1_macro", "retention_f1_macro"]);
writetable(T, fullfile(outDir, "vit_runs.csv"));

% ---- ViT groups next to the CNN summary (mean ± SD over seeds)
metrics = ["E1_accuracy", "E1_f1_macro", "E2_accuracy", "E2_f1_macro", "gap_f1_macro", "retention_f1_macro"];
groups = {"ViT-S/16 attention fine-tune", ["R1", "R4", "R5"]; ...
          "ViT-S/16 frozen backbone", "R2"; ...
          "ViT-S/16 attention + gray-world (ablation)", "R3"; ...
          "ViT-S/16 frozen backbone, LR 1e-3 (exploratory)", "R2x"; ...
          "ViT-S/16 linear probe, cached unaugmented features (exploratory)", "R2p"};
C = readtable(fullfile(cfg.repo, "shared", "results", "model_summary.csv"), ...
    'ReadRowNames', true, 'VariableNamingRule', 'preserve');
model = string(C.Properties.RowNames);
nSeeds = C.n_seeds;
mu = C{:, metrics + "_mean"};
sd = C{:, metrics + "_sd"};
for g = 1:size(groups, 1)
    k = ismember(T.run, groups{g, 2});
    if ~any(k), continue; end
    model(end + 1) = groups{g, 1}; %#ok<AGROW>
    nSeeds(end + 1) = nnz(k); %#ok<AGROW>
    mu(end + 1, :) = mean(T{k, metrics}, 1); %#ok<AGROW>
    if nnz(k) > 1, sd(end + 1, :) = std(T{k, metrics}, 0, 1); else, sd(end + 1, :) = NaN; end %#ok<AGROW>
end
S = [table(model, nSeeds, 'VariableNames', ["model", "n_seeds"]), ...
    array2table(mu, 'VariableNames', metrics + "_mean"), array2table(sd, 'VariableNames', metrics + "_sd")];
writetable(S, fullfile(outDir, "comparison.csv"));
fid = fopen(fullfile(outDir, "comparison.md"), "w");
fprintf(fid, "| Model | Seeds | E1 accuracy | E1 macro-F1 | E2 accuracy | E2 macro-F1 | Gap (E1−E2) | Retention |\n");
fprintf(fid, "|---|---|---|---|---|---|---|---|\n");
for i = 1:height(S)
    cells = arrayfun(@(j) meanSd(mu(i, j), sd(i, j)), 1:numel(metrics));
    fprintf(fid, "| %s | %d | %s |\n", model(i), nSeeds(i), strjoin(cells, " | "));
end
fprintf(fid, "\nMean ± SD over training seeds (single-seed rows: value only). CNN rows: shared/results/model_summary.csv (Teddy Ryu), unchanged.\n");
fclose(fid);

% ---- E2 per-class F1, ViT vs ResNet50
P = readtable(fullfile(cfg.repo, "shared", "results", "per_class_f1.csv"), ...
    'ReadRowNames', true, 'VariableNamingRule', 'preserve');
P = P(classNames, :);                          % mapping order, by name
bars = []; legendText = strings(0);
for pair = ["R1", "ResNet50 full fine-tune E2"; "R2", "ResNet50 frozen backbone E2"]'
    if isfield(R, pair(1))
        pc = readtable(fullfile(runDir(pair(1)), "per_class_external.csv"), 'ReadRowNames', true);
        bars(:, end + 1) = pc{classNames, "f1"}; %#ok<AGROW>
        legendText(end + 1) = "ViT " + pair(1) + " (" + R.(pair(1)).run.regime + ")"; %#ok<AGROW>
    end
    bars(:, end + 1) = P{:, pair(2)}; %#ok<AGROW>
    legendText(end + 1) = erase(pair(2), " E2") + " (mean of 3 seeds)"; %#ok<AGROW>
end
for extra = ["R2x", "ViT R2x (frozen, LR 1e-3, exploratory)"; "R2p", "ViT R2p (linear probe, exploratory)"]'
    if ~isfield(R, extra(1)), continue; end
    pc = readtable(fullfile(runDir(extra(1)), "per_class_external.csv"), 'ReadRowNames', true);
    bars(:, end + 1) = pc{classNames, "f1"}; %#ok<AGROW>
    legendText(end + 1) = extra(2); %#ok<AGROW>
end
fig = figure("Visible", "off", "Theme", "light", "Position", [100 100 900 650]);
barh(categorical(classNames, flip(classNames)), bars);
xlabel("F1 on Cropped-PlantDoc (E2)"); xlim([0 1]); grid on;
legend(legendText, "Location", "southeast", "Interpreter", "none");
set(gca, "TickLabelInterpreter", "none");
exportgraphics(fig, fullfile(outDir, "per_class_f1_E2.png"), "Resolution", 200);
close(fig);

% ---- training loss
fig = figure("Visible", "off", "Theme", "light", "Position", [100 100 900 450]);
hold on;
for id = setdiff(string(T.run), "R2p", "stable")'   % R2p trains on cached features, a different iteration scale
    H = readtable(fullfile(runDir(id), "training_history.csv"));
    plot(H.Iteration, movmean(H.Loss, 100), "DisplayName", id + " (" + R.(id).run.regime + ")");
end
hold off; grid on; set(gca, "YScale", "log");
xlabel("Iteration (1,014 per epoch)"); ylabel("Training loss (moving average, 100 iterations)");
legend("Location", "northeast");
exportgraphics(fig, fullfile(outDir, "training_loss.png"), "Resolution", 200);
close(fig);

% ---- fixed qualitative sample, R1 predictions
if isfield(R, "R1")
    G = readtable(fullfile(cfg.repo, "shared", "gradcam_sample.csv"), "TextType", "string");
    Q = readtable(fullfile(runDir("R1"), "predictions_external.csv"), "TextType", "string");
    [found, loc] = ismember(G.image_id, Q.image_id);
    assert(all(found), "gradcam_sample.csv has images missing from R1 external predictions");
    [M] = loadManifest(cfg);
    [~, mloc] = ismember(G.image_id, M.image_id);
    fig = figure("Visible", "off", "Theme", "light", "Position", [50 50 1500 1100]);
    tl = tiledlayout(4, 6, "TileSpacing", "compact", "Padding", "compact");
    for i = 1:height(G)
        nexttile;
        imshow(preprocessImage(M.Path(mloc(i)), "eval", Config=cfg));
        yp = Q.y_pred(loc(i));
        ok = yp == G.y(i);
        title(sprintf("true: %s\npred: %s", classNames(G.y(i) + 1), classNames(yp + 1)), ...
            "Interpreter", "none", "FontSize", 7, "Color", [~ok, 0.5 * ok, 0]);
    end
    title(tl, "R1 on the fixed external sample (2 per class); green = correct, red = wrong", "FontSize", 11);
    exportgraphics(fig, fullfile(outDir, "examples_R1.png"), "Resolution", 150);
    close(fig);
end
% ---- confusion matrices redrawn for the report (light theme; evaluateSplit's PNGs follow the desktop theme)
for id = string(T.run)'
    for split = ["internal", "external"]
        K = readtable(fullfile(runDir(id), "confusion_" + split + ".csv"), 'ReadRowNames', true);
        fig = figure("Visible", "off", "Theme", "light", "Position", [100 100 950 850]);
        labels = replace(classNames, "_", " ");   % confusionchart has no Interpreter: "_" would subscript
        cm = confusionchart(K{:, :}, labels, "Normalization", "row-normalized", ...
            "Title", sprintf("%s, %s: row-normalised (macro-F1 %.3f)", id, ...
            replace(split, ["internal", "external"], ["E1 PlantVillage test", "E2 Cropped-PlantDoc"]), ...
            R.(id).(replace(split, ["internal", "external"], ["E1", "E2"])).f1Macro));
        sortClasses(cm, labels);                  % mapping order, not alphabetical
        exportgraphics(fig, fullfile(outDir, "confusion_" + id + "_" + split + ".png"), "Resolution", 200);
        close(fig);
    end
end
fprintf("Summary of %d run(s) written to %s\n", height(T), outDir);
end

function s = meanSd(m, sd)
if isnan(sd), s = sprintf("%.3f", m); else, s = sprintf("%.3f ± %.3f", m, sd); end
end
