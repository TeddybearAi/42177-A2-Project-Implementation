function scores = predictScores(net, paths, grayWorld, cfg)
%PREDICTSCORES Class probabilities (N x 12, columns in class_mapping order) for image files.
%   Uses the deterministic evaluation transform (shorter side 438, centre crop 384), with
%   gray-world white balance first when the run uses it. Row order = order of paths.
ds = imageDatastore(paths, "ReadFcn", @(f) preprocessImage(f, "eval", GrayWorld=grayWorld, Config=cfg));
scores = gather(minibatchpredict(net, ds, "MiniBatchSize", 32));
assert(isequal(size(scores), [numel(paths), 12]), "unexpected score size %s", mat2str(size(scores)));
end
