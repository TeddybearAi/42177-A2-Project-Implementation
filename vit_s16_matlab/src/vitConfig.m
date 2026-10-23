function cfg = vitConfig()
%VITCONFIG Paths and fixed settings of the ViT workstream (protocol: docs/adr/0002).
%   Dataset folders default to PlantVillage/ and PlantDoc-Dataset/ next to the
%   repository folder; edit pvRoot / pdRoot here if your copies live elsewhere.

cfg.repo = fileparts(fileparts(mfilename("fullpath")));
cfg.pvRoot = fullfile(fileparts(cfg.repo), "PlantVillage");      % contains Pepper__bell___Bacterial_spot, ...
cfg.pdRoot = fullfile(fileparts(cfg.repo), "PlantDoc-Dataset");  % contains train/ and test/
cfg.manifest = fullfile(cfg.repo, "shared", "manifest.csv");
cfg.mapping = fullfile(cfg.repo, "shared", "class_mapping.json");
cfg.outDir = fullfile(cfg.repo, "outputs");

% ---- preprocessing (ADR 0002)
cfg.inputSize = 384;                        % visionTransformer "*-16-imagenet-384" input
cfg.resizeShort = floor(384 * 256 / 224);   % 438: shorter side before the centre crop (CNNs' ratio)

% ---- training-only augmentation, mirroring the CNNs' torchvision settings
cfg.aug.cropArea = [0.6 1.0];               % random resized crop: share of the image area
cfg.aug.cropRatio = [3/4 4/3];              % ... and width/height ratio
cfg.aug.flipProb = 0.5;                     % horizontal and vertical, independently
cfg.aug.maxRotation = 20;                   % degrees, uniform in [-20, 20], black fill
cfg.aug.jitter = 0.3;                       % brightness/contrast/saturation factor in [0.7, 1.3]
cfg.aug.hue = 0.05;                         % hue shift in [-0.05, 0.05] (fraction of the colour wheel)

% ---- reference partition sizes from the shared manifest (audit compares against these)
cfg.expectedCounts = struct("train", 12174, "val", 2609, "test", 2609, "external", 1097);

% ---- model and training recipe (MathWorks ViT transfer-learning example; ADR 0001/0002)
cfg.model = "small-16-imagenet-384";
cfg.learnRate = 1e-4;                       % Adam, constant
cfg.miniBatchSize = 12;                     % fallback: 6 if out of GPU memory (ADR 0002)
cfg.maxEpochs = 4;
cfg.nBoot = 1000;                           % bootstrap resamples for the macro-F1 CI
cfg.preprocessWorkers = 2;                  % CPU workers for augmentation: 2 keep up with the GPU
                                            % (~0.6 s/batch each vs 1.6 s/iteration) without exhausting 16 GB RAM

% ---- runs, in the pre-specified order (ADR 0002); R2x is exploratory, added after R2's result
cfg.runs = struct( ...
    "id",        {"R1",        "R2",     "R3",        "R4",        "R5",        "R2x"}, ...
    "regime",    {"attention", "frozen", "attention", "attention", "attention", "frozen"}, ...
    "grayWorld", {false,       false,    true,        false,       false,       false}, ...
    "seed",      {42,          42,       42,          43,          44,          42}, ...
    "learnRate", {NaN,         NaN,      NaN,         NaN,         NaN,         1e-3});   % NaN = cfg.learnRate
end
