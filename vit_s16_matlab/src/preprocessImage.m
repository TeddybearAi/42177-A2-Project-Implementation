function I = preprocessImage(file, mode, opts)
%PREPROCESSIMAGE Decode one image and apply the protocol's preprocessing (ADR 0002).
%   I = preprocessImage(file, "decode")  3-channel uint8 RGB at the original size
%   I = preprocessImage(file, "eval")    deterministic 384x384x3 uint8: shorter side to 438,
%                                        centre crop (val, test and external partitions)
%   I = preprocessImage(file, "train")   randomly augmented 384x384x3 uint8 (train only)
%   ... = preprocessImage(..., GrayWorld=true) white-balances the image first (run R3).
%   Output stays in [0, 255]: the network's own input layer does the z-score normalisation.
arguments
    file (1,1) string
    mode (1,1) string {mustBeMember(mode, ["decode", "eval", "train"])}
    opts.GrayWorld (1,1) logical = false
    opts.Config struct = vitConfig()
end
cfg = opts.Config;

I = readRGB(file);
if opts.GrayWorld
    I = grayWorld(I);
end
switch mode
    case "eval"
        I = centreView(I, cfg.resizeShort, cfg.inputSize);
    case "train"
        I = augmentedView(I, cfg.inputSize, cfg.aug);
end
end

% =========================================================================== decoding
function I = readRGB(file)
% Any supported file -> uint8 RGB. Palette images use their colour map; greyscale is
% replicated to 3 channels; 16-bit data is rescaled; an alpha channel is dropped
% (imread returns it separately). 4-channel (CMYK) data is rejected rather than guessed.
[I, map] = imread(file);
if ~isempty(map)
    I = ind2rgb(I, map);
end
I = im2uint8(I);
if size(I, 3) == 1
    I = repmat(I, [1 1 3]);
elseif size(I, 3) ~= 3
    error("preprocessImage:channels", "%s has %d channels; expected 1 or 3.", file, size(I, 3));
end
end

function I = grayWorld(I)
% Gray-world colour constancy: scale each channel so all three have the image's mean grey level.
X = single(I);
channelMean = mean(X, [1 2]);
gain = mean(channelMean) ./ max(channelMean, 1);   % guard against an all-black channel
I = uint8(min(X .* gain, 255));
end

% =========================================================================== geometry
function I = centreView(I, shortSide, cropSize)
% Shorter side to shortSide (aspect ratio kept, as torchvision Resize(int)), then centre crop.
[h, w, ~] = size(I);
if h <= w
    sz = [shortSide, floor(shortSide * w / h)];
else
    sz = [floor(shortSide * h / w), shortSide];
end
I = imresize(I, sz, "bilinear", "Antialiasing", true);
top = round((sz(1) - cropSize) / 2);
left = round((sz(2) - cropSize) / 2);
I = I(top + 1:top + cropSize, left + 1:left + cropSize, :);
end

function I = augmentedView(I, outSize, aug)
% Random resized crop -> flips -> rotation -> colour jitter, as torchvision's training transform.
I = randomResizedCrop(I, outSize, aug.cropArea, aug.cropRatio);
if rand < aug.flipProb, I = fliplr(I); end
if rand < aug.flipProb, I = flipud(I); end
I = imrotate(I, (2 * rand - 1) * aug.maxRotation, "bilinear", "crop");   % keeps size, black fill
I = colourJitter(I, aug.jitter, aug.hue);
end

function I = randomResizedCrop(I, outSize, areaRange, ratioRange)
% torchvision.transforms.RandomResizedCrop: up to 10 tries at a random area share and
% (log-uniform) aspect ratio; falls back to a centre crop with the ratio clamped to range.
[h, w, ~] = size(I);
area = h * w;
for attempt = 1:10
    targetArea = area * (areaRange(1) + rand * diff(areaRange));
    ratio = exp(log(ratioRange(1)) + rand * diff(log(ratioRange)));
    cw = round(sqrt(targetArea * ratio));
    ch = round(sqrt(targetArea / ratio));
    if cw >= 1 && ch >= 1 && cw <= w && ch <= h
        top = randi([0, h - ch]);
        left = randi([0, w - cw]);
        I = imresize(I(top + 1:top + ch, left + 1:left + cw, :), [outSize outSize], ...
            "bilinear", "Antialiasing", true);
        return
    end
end
ratio = w / h;
if ratio < ratioRange(1)
    cw = w; ch = round(w / ratioRange(1));
elseif ratio > ratioRange(2)
    ch = h; cw = round(h * ratioRange(2));
else
    cw = w; ch = h;
end
top = floor((h - ch) / 2);
left = floor((w - cw) / 2);
I = imresize(I(top + 1:top + ch, left + 1:left + cw, :), [outSize outSize], ...
    "bilinear", "Antialiasing", true);
end

% =========================================================================== colour
function I = colourJitter(I, strength, maxHue)
% torchvision ColorJitter(b, c, s, h): the four adjustments in random order, each clipped to
% [0, 1]. Brightness scales, contrast blends with the mean grey level, saturation blends
% with the greyscale image, hue rotates the HSV hue channel.
X = im2single(I);
for k = randperm(4)
    switch k
        case 1
            X = X * factor(strength);
        case 2
            m = mean(greyLevel(X), "all");
            X = (X - m) * factor(strength) + m;
        case 3
            g = greyLevel(X);
            X = (X - g) * factor(strength) + g;
        case 4
            hsv = rgb2hsv(X);
            hsv(:, :, 1) = mod(hsv(:, :, 1) + (2 * rand - 1) * maxHue, 1);
            X = hsv2rgb(hsv);
    end
    X = min(max(X, 0), 1);
end
I = im2uint8(X);
end

function f = factor(strength)
f = 1 + (2 * rand - 1) * strength;   % uniform in [1 - strength, 1 + strength]
end

function g = greyLevel(X)
g = 0.299 * X(:, :, 1) + 0.587 * X(:, :, 2) + 0.114 * X(:, :, 3);   % same weights as torchvision
end
