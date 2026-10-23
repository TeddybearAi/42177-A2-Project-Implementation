%SELFTEST Self-check of loadManifest, preprocessImage and the metric functions (run from src/).
%   matlab -batch "cd src; selfTest"
% Uses small synthetic images plus the shared manifest; every check is an assert.

cfg = vitConfig();
tmp = tempname; mkdir(tmp);
cleanup = onCleanup(@() rmdir(tmp, "s"));

% ---- decoding: greyscale, palette and 16-bit files all become uint8 RGB
grey = uint8(repmat(linspace(0, 255, 500), 300, 1));
imwrite(grey, fullfile(tmp, "grey.png"));
I = preprocessImage(fullfile(tmp, "grey.png"), "decode");
assert(isequal(size(I), [300 500 3]) && isa(I, "uint8") && isequal(I(:, :, 1), I(:, :, 3)));

map = [1 0 0; 0 1 0; 0 0 1];
imwrite(uint8([0 1; 2 0]), map, fullfile(tmp, "palette.png"));
I = preprocessImage(fullfile(tmp, "palette.png"), "decode");
assert(isequal(squeeze(I(1, 2, :))', uint8([0 255 0])), "palette colour lost");

imwrite(uint16(65535 * ones(10, 10, 3)), fullfile(tmp, "deep.png"));
I = preprocessImage(fullfile(tmp, "deep.png"), "decode");
assert(isa(I, "uint8") && all(I(:) == 255), "16-bit image not rescaled");

% ---- eval: 384x384, deterministic, aspect ratio kept (centre of a wide image survives)
wide = zeros(200, 800, 3, "uint8");
wide(:, 1:300, 1) = 255;          % red band on the left edge, cut off by the centre crop
wide(:, 301:500, 2) = 255;        % green band in the middle
imwrite(wide, fullfile(tmp, "wide.png"));
A = preprocessImage(fullfile(tmp, "wide.png"), "eval");
B = preprocessImage(fullfile(tmp, "wide.png"), "eval");
assert(isequal(size(A), [384 384 3]) && isa(A, "uint8") && isequal(A, B));
assert(A(192, 192, 2) == 255 && A(192, 192, 1) == 0, "centre crop is not centred");
assert(cfg.resizeShort == 438);

% ---- train: 384x384, random, reproducible under the same seed
rng(1); T1 = preprocessImage(fullfile(tmp, "wide.png"), "train");
rng(1); T2 = preprocessImage(fullfile(tmp, "wide.png"), "train");
T3 = preprocessImage(fullfile(tmp, "wide.png"), "train");
assert(isequal(size(T1), [384 384 3]) && isa(T1, "uint8"));
assert(isequal(T1, T2), "augmentation not reproducible with rng");
assert(~isequal(T1, T3), "augmentation is not random");

% ---- gray-world: a colour cast is removed (channel means become equal)
cast = uint8(cat(3, 200 * ones(50), 100 * ones(50), 50 * ones(50)));
imwrite(cast, fullfile(tmp, "cast.png"));
G = single(preprocessImage(fullfile(tmp, "cast.png"), "decode", GrayWorld=true));
m = squeeze(mean(G, [1 2]));
assert(max(m) - min(m) <= 1, "gray-world did not equalise channel means");

% ---- manifest: explicit class order, shared counts, no alphabetical reordering
[M, classNames] = loadManifest(cfg);
assert(height(M) == 18489 && numel(classNames) == 12);
assert(isequal(string(categories(M.Y))', classNames), "category order differs from mapping");
assert(classNames(1) == "Pepper_bell_bacterial_spot" && classNames(12) == "Tomato_healthy");
assert(all(double(M.Y) == M.y + 1), "categorical index does not equal y + 1");
for p = ["train", "val", "test", "external"]
    assert(nnz(M.split == p) == cfg.expectedCounts.(p), "count mismatch in " + p);
end

% ---- metrics: hand-computed example (class 3 never predicted -> F1 = 0, kept in the macro mean)
m = classificationMetrics([1 1 2 2], [1 2 2 2], 3);
assert(abs(m.accuracy - 0.75) < 1e-12);
assert(all(abs(m.perClass.f1 - [2/3; 0.8; 0]) < 1e-12));
assert(abs(m.f1Macro - (2/3 + 0.8) / 3) < 1e-12);
assert(isequal(m.confusion, [1 1 0; 0 2 0; 0 0 0]));
ci = bootstrapF1([1 1 2 2 3 3], [1 2 2 2 3 1], 3, 7, 200);
assert(isequal(ci, bootstrapF1([1 1 2 2 3 3], [1 2 2 2 3 1], 3, 7, 200)) && ci(1) <= ci(2));

disp("selfTest: all checks passed");
