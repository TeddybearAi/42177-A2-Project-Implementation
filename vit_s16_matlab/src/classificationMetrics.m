function m = classificationMetrics(yTrue, yPred, nClasses)
%CLASSIFICATIONMETRICS Accuracy, macro and per-class precision/recall/F1, confusion matrix.
%   yTrue, yPred are class indices 1..nClasses (MATLAB index = shared y + 1).
%   Every class is always included, so a class that is never predicted counts as F1 = 0
%   instead of silently leaving the macro average; 0/0 is scored 0 (as scikit-learn's
%   zero_division=0, which the group's CNN results use).
yTrue = yTrue(:); yPred = yPred(:);
C = accumarray([yTrue, yPred], 1, [nClasses, nClasses]);   % rows = true class, columns = predicted
tp = diag(C);
predicted = sum(C, 1)';
actual = sum(C, 2);
precision = safeDivide(tp, predicted);
recall = safeDivide(tp, actual);
f1 = safeDivide(2 * tp, predicted + actual);

m.n = numel(yTrue);
m.accuracy = sum(tp) / m.n;
m.precisionMacro = mean(precision);
m.recallMacro = mean(recall);
m.f1Macro = mean(f1);
m.perClass = table(precision, recall, f1, actual, 'VariableNames', ["precision", "recall", "f1", "support"]);
m.confusion = C;
end

function r = safeDivide(a, b)
r = zeros(size(a));
r(b > 0) = a(b > 0) ./ b(b > 0);
end
