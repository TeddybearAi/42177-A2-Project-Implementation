function summary = evaluateSplit(net, rows, classNames, name, title, outDir, run, cfg)
%EVALUATESPLIT Locked evaluation of the selected network on one partition (E1 or E2).
%   Writes predictions_<name>.csv (shared format: image_id, y_true, y_pred, p_0..p_11 with
%   0-based class indices), per_class_<name>.csv, confusion_<name>.csv/.png, and returns
%   accuracy, macro precision/recall/F1 and a 95% bootstrap CI for macro-F1.
nClasses = numel(classNames);
scores = predictScores(net, rows.Path, run.grayWorld, cfg);
[~, yPred] = max(scores, [], 2);
yTrue = double(rows.Y);                       % category k <-> shared y = k - 1 (loadManifest)
m = classificationMetrics(yTrue, yPred, nClasses);

% ---- predictions in the group's shared format, one row per manifest image
P = table(rows.image_id, yTrue - 1, yPred - 1, 'VariableNames', ["image_id", "y_true", "y_pred"]);
P = [P, array2table(scores, 'VariableNames', "p_" + string(0:nClasses - 1))];
assert(height(P) == numel(unique(P.image_id)) && all(P.y_true == rows.y), "prediction rows do not match manifest");
writetable(P, fullfile(outDir, "predictions_" + name + ".csv"));

perClass = m.perClass;
perClass.Properties.RowNames = classNames;
writetable(perClass, fullfile(outDir, "per_class_" + name + ".csv"), 'WriteRowNames', true);
writetable(array2table(m.confusion, 'VariableNames', classNames, 'RowNames', classNames), ...
    fullfile(outDir, "confusion_" + name + ".csv"), 'WriteRowNames', true);

fig = figure("Visible", "off", "Position", [100 100 950 850]);
confusionchart(m.confusion, classNames, "Normalization", "row-normalized", ...
    "Title", sprintf("%s (%s, %s regime, seed %d) - %s, macro-F1 %.3f", ...
    run.id, cfg.model, run.regime, run.seed, title, m.f1Macro));
exportgraphics(fig, fullfile(outDir, "confusion_" + name + ".png"), "Resolution", 200);
close(fig);

summary = rmfield(m, ["perClass", "confusion"]);
summary.f1MacroCI95 = bootstrapF1(yTrue, yPred, nClasses, run.seed, cfg.nBoot);
end
