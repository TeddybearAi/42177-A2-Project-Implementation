function ci = bootstrapF1(yTrue, yPred, nClasses, seed, nBoot)
%BOOTSTRAPF1 95% percentile bootstrap interval for macro-F1 (Efron & Tibshirani 1993).
%   Resamples the evaluated images with replacement nBoot times (default 1000) and
%   recomputes macro-F1; returns [2.5th, 97.5th] percentiles. Seeded, so repeatable.
%   It reflects uncertainty from the finite test set only, not from training randomness.
arguments
    yTrue
    yPred
    nClasses
    seed = 0
    nBoot = 1000
end
stream = RandStream("mt19937ar", "Seed", seed);   % private stream: leaves the global rng untouched
n = numel(yTrue);
stats = zeros(nBoot, 1);
for b = 1:nBoot
    idx = randi(stream, n, n, 1);
    m = classificationMetrics(yTrue(idx), yPred(idx), nClasses);
    stats(b) = m.f1Macro;
end
sorted = sort(stats);
ci = [sorted(max(1, round(0.025 * nBoot))), sorted(round(0.975 * nBoot))];
end
