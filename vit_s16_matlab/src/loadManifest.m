function [M, classNames, mapping] = loadManifest(cfg)
%LOADMANIFEST Read the shared manifest and class mapping, with explicit label order.
%   M has the manifest columns plus Path (absolute file path) and Y (categorical
%   whose category order is the class_mapping.json order, so category k <-> y = k-1).
%   MATLAB's default alphabetical category order is never used. Errors if a row's
%   label and index disagree with the mapping, since every result would then be wrong.
arguments
    cfg = vitConfig()
end

opts = detectImportOptions(cfg.manifest, "Delimiter", ",", "TextType", "string");
textCols = ["image_id", "root", "relpath", "label", "source", "md5", "split"];
missingCols = setdiff([textCols, "y"], string(opts.VariableNames));
if ~isempty(missingCols)
    error("Manifest %s lacks columns: %s", cfg.manifest, strjoin(missingCols, ", "));
end
opts = setvartype(opts, textCols, "string");    % md5 / ids must never be parsed as numbers
opts = setvartype(opts, "y", "double");
M = readtable(cfg.manifest, opts);

mapping = jsondecode(fileread(cfg.mapping));
classNames = string({mapping.classes.label});
cn = classNames(:);

% ---- label contract: y is 0..11 and names the same class as the label column
validY = M.y == round(M.y) & M.y >= 0 & M.y < numel(cn);
if ~all(validY)
    error("%d manifest rows have y outside 0..%d, e.g. %s", nnz(~validY), numel(cn) - 1, ...
        M.image_id(find(~validY, 1)));
end
wrong = M.label ~= cn(M.y + 1);
if any(wrong)
    error("%d manifest rows have a label that does not match class_mapping.json index y, e.g. %s", ...
        nnz(wrong), M.image_id(find(wrong, 1)));
end

% ---- absolute paths ("pv" = PlantVillage root, "pd" = Cropped-PlantDoc root)
M.Path = strings(height(M), 1);
isPv = M.root == "pv";
isPd = M.root == "pd";
if ~all(isPv | isPd)
    error("Unknown manifest root %s", M.root(find(~(isPv | isPd), 1)));
end
M.Path(isPv) = fullfile(cfg.pvRoot, M.relpath(isPv));
M.Path(isPd) = fullfile(cfg.pdRoot, M.relpath(isPd));

M.Y = categorical(M.label, classNames);   % explicit category order = mapping order
end
