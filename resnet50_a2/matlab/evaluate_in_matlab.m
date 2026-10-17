%% evaluate_in_matlab.m
% Reproduces the evaluation of the trained ResNet50 and EfficientNet-B0 models in MATLAB.
% 42177 Image Processing and Pattern Recognition, A2, Group 2. Author: Teddy Ryu.
%
% The models were trained in Python/PyTorch (train.py) and exported to ONNX with
% export_onnx.py. This script imports them, classifies the PlantDoc external test
% images (or the PlantVillage internal test images), and reports accuracy, macro-F1,
% per-class F1 and a row-normalised confusion matrix. It then compares its predictions
% with the Python predictions saved in runs/<run>/predictions_external.csv.
%
% Requirements
%   MATLAB R2023b or later, Deep Learning Toolbox, Image Processing Toolbox, and the free
%   add-on "Deep Learning Toolbox Converter for ONNX Model Format" (Home > Add-Ons).
%   Older releases (R2020b-R2023a) are also handled, using importONNXNetwork instead.
%
% How to run
%   1. In Python (once):  python export_onnx.py      -> creates matlab/*.onnx
%   2. Open this file in MATLAB and press Run. PlantDoc is downloaded automatically if it is
%      not found next to the project folder (or set pdRoot below to your own copy).
%   Tested in MATLAB Online: the external macro-F1 matched Python to within 0.002.
%
% Outputs (in this matlab folder)
%   <run>_predictions_<split>.csv, <run>_confusion_<split>.png, matlab_summary_<split>.csv

clear; clc;

%% ------------------------------------------------------------ project files
matlabDir = fileparts(mfilename("fullpath"));        % this folder (project/matlab)
projectDir = fileparts(matlabDir);                   % project folder
manifestFile = fullfile(projectDir, "outputs", "manifest.csv");
mappingFile = fullfile(projectDir, "class_mapping.json");
runsDir = fullfile(projectDir, "runs");

%% ------------------------------------------------------------ settings
% Dataset folders. By default they sit next to the project folder; PlantDoc is
% downloaded there automatically from GitHub if it is missing (about 1 GB, once).
pdRoot = fullfile(fileparts(projectDir), "PlantDoc-Dataset-master");   % folder with PlantDoc train/ and test/
pvRoot = fullfile(fileparts(projectDir), "PlantVillage");              % only needed when evalSplit = "test"
runs = ["full_s44", "effb0_full_s44"];   % models exported by export_onnx.py
evalSplit = "external";                  % "external" = PlantDoc (E2); "test" = PlantVillage internal test (E1)
batchSize = 32;                          % images per prediction call

if evalSplit == "external" && ~isfolder(pdRoot)
    fprintf("PlantDoc not found at %s\nDownloading it from GitHub (about 1 GB, one time only)...\n", pdRoot);
    unzip("https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip", fileparts(pdRoot));
end

% Class names in the training order (index 0..11 in Python = 1..12 here)
mapping = jsondecode(fileread(mappingFile));
classNames = string({mapping.classes.label});
nClasses = numel(classNames);

% Shared manifest: the same image list and split used by every group model
opts = detectImportOptions(manifestFile, "Delimiter", ",");
opts = setvartype(opts, ["image_id", "root", "relpath", "label", "source", "split"], "string");
M = readtable(manifestFile, opts);
M = M(M.split == evalSplit, :);
nImages = height(M);
fprintf("%d images in split '%s'\n", nImages, evalSplit);

% Absolute image paths ("pv" = PlantVillage root, "pd" = PlantDoc root)
paths = strings(nImages, 1);
isPd = M.root == "pd";
paths(isPd) = fullfile(pdRoot, M.relpath(isPd));
paths(~isPd) = fullfile(pvRoot, M.relpath(~isPd));
missing = ~isfile(paths);
if any(missing)                                      % names changed by Windows extraction (see resolveRenamed)
    paths(missing) = resolveRenamed(paths(missing));
    missing = ~isfile(paths);
end
if any(missing)
    error("%d image files not found, e.g. %s - check pvRoot and pdRoot.", nnz(missing), paths(find(missing, 1)));
end
yTrue = M.y + 1;                                     % MATLAB class indices are 1-based

%% ------------------------------------------------------------ preprocess all images once
% Same steps as the Python evaluation transform: shorter side resized to 256 px,
% centre crop to 224 x 224, RGB scaled to [0, 1]. (ImageNet normalisation is inside the ONNX model.)
fprintf("Preprocessing images...\n");
X = zeros(224, 224, 3, nImages, "single");
for i = 1:nImages
    X(:, :, :, i) = preprocessImage(paths(i));
end

%% ------------------------------------------------------------ evaluate each model
useNewImporter = exist("importNetworkFromONNX", "file") == 2;
pySet = "external"; if evalSplit == "test", pySet = "internal"; end
summary = table();

for r = runs
    onnxFile = fullfile(matlabDir, r + ".onnx");
    if ~isfile(onnxFile)
        error("%s not found - run 'python export_onnx.py' first.", onnxFile);
    end
    fprintf("\n=== %s ===\n", r);

    % ---- import the network
    if useNewImporter
        % ONNX images are batch x channel x height x width; telling the importer this
        % ("BCSS") gives an image input layer that accepts MATLAB "SSCB" image batches.
        net = importNetworkFromONNX(onnxFile, ...
            "InputDataFormats", "BCSS", "OutputDataFormats", "BC");
        if ~net.Initialized
            net = initialize(net, dlarray(zeros(224, 224, 3, 1, "single"), "SSCB"));
        end
    else
        net = importONNXNetwork(onnxFile, "OutputLayerType", "classification", ...
            "Classes", categorical(classNames));        % DAGNetwork (older releases)
    end

    % ---- predict in batches
    yPred = zeros(nImages, 1);
    for s = 1:batchSize:nImages
        idx = s:min(s + batchSize - 1, nImages);
        if useNewImporter
            scores = extractdata(predict(net, dlarray(X(:, :, :, idx), "SSCB")));
            if size(scores, 1) ~= nClasses, scores = scores.'; end   % make it classes x images
            [~, k] = max(scores, [], 1);
        else
            scores = predict(net, X(:, :, :, idx));                  % images x classes
            [~, k] = max(scores, [], 2);
        end
        yPred(idx) = k(:);
    end

    % ---- metrics (computed directly, no extra toolbox needed)
    C = accumarray([yTrue, yPred], 1, [nClasses, nClasses]);        % rows = true, columns = predicted
    [acc, macroF1, f1] = classificationMetrics(C);
    fprintf("accuracy %.4f | macro-F1 %.4f\n", acc, macroF1);
    disp(table(classNames(:), f1, sum(C, 2), 'VariableNames', {'class', 'F1', 'support'}));

    % ---- compare with the Python predictions for the same images
    pyFile = fullfile(runsDir, r, "predictions_" + pySet + ".csv");
    agreement = NaN; pyMacroF1 = NaN;
    if isfile(pyFile)
        P = readtable(pyFile, "TextType", "string");
        [found, loc] = ismember(M.image_id, P.image_id);
        agreement = mean(yPred(found) == P.y_pred(loc(found)) + 1);
        Cpy = accumarray([P.y_true + 1, P.y_pred + 1], 1, [nClasses, nClasses]);
        [~, pyMacroF1] = classificationMetrics(Cpy);
        fprintf("Python macro-F1 %.4f | prediction agreement MATLAB vs Python %.1f%% (%d images)\n", ...
            pyMacroF1, 100 * agreement, nnz(found));
    else
        fprintf("(no Python predictions found at %s)\n", pyFile);
    end

    % ---- save predictions and a confusion matrix figure
    T = table(M.image_id, M.y, yPred - 1, 'VariableNames', {'image_id', 'y_true', 'y_pred'});
    writetable(T, fullfile(matlabDir, r + "_predictions_" + evalSplit + ".csv"));
    fig = figure("Visible", "off", "Position", [100 100 950 850]);
    confusionchart(C, classNames, "Normalization", "row-normalized", ...
        "Title", r + " (MATLAB) - " + evalSplit + " test, macro-F1 " + sprintf("%.3f", macroF1));
    exportgraphics(fig, fullfile(matlabDir, r + "_confusion_" + evalSplit + ".png"), "Resolution", 200);
    close(fig);

    summary = [summary; table(r, acc, macroF1, pyMacroF1, agreement, ...
        'VariableNames', {'run', 'accuracy', 'macro_F1_matlab', 'macro_F1_python', 'prediction_agreement'})]; %#ok<AGROW>
end

disp(summary);
writetable(summary, fullfile(matlabDir, "matlab_summary_" + evalSplit + ".csv"));

%% ------------------------------------------------------------ local functions
function X = preprocessImage(file)
% Read one image and apply the evaluation transform used in Python.
    [I, map] = imread(file);
    if ~isempty(map), I = ind2rgb(I, map); end            % indexed (palette) images
    if size(I, 3) == 1, I = repmat(I, [1 1 3]); end        % greyscale -> RGB
    if size(I, 3) > 3, I = I(:, :, 1:3); end               % drop any extra channel
    I = im2single(I);                                      % scale to [0, 1]
    [h, w, ~] = size(I);
    if h <= w                                              % shorter side -> 256 px, keep aspect ratio
        newH = 256; newW = floor(256 * w / h);
    else
        newW = 256; newH = floor(256 * h / w);
    end
    I = imresize(I, [newH, newW], "bilinear", "Antialiasing", true);
    top = round((newH - 224) / 2); left = round((newW - 224) / 2);
    X = I(top + 1:top + 224, left + 1:left + 224, :);      % centre crop 224 x 224
end

function [acc, macroF1, f1] = classificationMetrics(C)
% Accuracy, macro-F1 and per-class F1 from a confusion matrix (rows = true classes).
% Matches scikit-learn's macro F1 with zero_division = 0.
    tp = diag(C);
    fp = sum(C, 1).' - tp;
    fn = sum(C, 2) - tp;
    denom = 2 * tp + fp + fn;
    f1 = zeros(size(tp));
    f1(denom > 0) = 2 * tp(denom > 0) ./ denom(denom > 0);
    acc = sum(tp) / sum(C(:));
    macroF1 = mean(f1);
end

function out = resolveRenamed(in)
% Windows cannot store the characters < > : " | ? * in file names, so a dataset extracted on
% Windows has them replaced by "_" (for example "file.php?id=5336&t=1.jpg" becomes
% "file.php_id=5336&t=1.jpg"), and the shared manifest was written on Windows. On Linux and
% macOS the original names are kept. For each missing file, look in its folder for a file
% whose name, after the same replacement, matches the manifest name.
    out = in;
    for k = 1:numel(in)
        [folder, name, ext] = fileparts(in(k));
        if ~isfolder(folder), continue; end
        listing = dir(folder);
        listing = listing(~[listing.isdir]);
        if isempty(listing), continue; end
        candidates = string({listing.name});
        sanitised = regexprep(candidates, '[<>:"|?*]', '_');
        hit = find(sanitised == name + ext, 1);
        if ~isempty(hit)
            out(k) = fullfile(folder, candidates(hit));
        end
    end
end
