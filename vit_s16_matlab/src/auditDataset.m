function summary = auditDataset(cfg)
%AUDITDATASET Independent check of the shared manifest against the local image files.
%   For every manifest row: file present, MD5 equal to the manifest's, decodable to RGB
%   by preprocessImage, format, size and colour type, and class folder consistent with
%   class_mapping.json. Across rows: partition and class counts against the reference
%   counts, exact duplicates (MD5) within and across partitions, and PlantVillage <->
%   PlantDoc overlap. MD5 only detects byte-identical files; near-duplicates are a
%   separate, unchecked limitation.
%   Writes outputs/audit/audit_rows.csv (one row per image, with a note for every
%   problem), counts.csv and audit_summary.json. Nothing is skipped or fixed silently.
arguments
    cfg = vitConfig()
end

[M, classNames, mapping] = loadManifest(cfg);
n = height(M);
outDir = fullfile(cfg.outDir, "audit");
if ~isfolder(outDir), mkdir(outDir); end

% ---- per-image checks
exists = isfile(M.Path);
md5ok = false(n, 1); readable = false(n, 1); folderOk = false(n, 1);
format = strings(n, 1); colourType = strings(n, 1); imgWidth = nan(n, 1); imgHeight = nan(n, 1);
note = strings(n, 1);
hasher = java.security.MessageDigest.getInstance("MD5");
pvFolders = normaliseName(string({mapping.classes.plantvillage}));
pdFolders = string({mapping.classes.plantdoc});
fprintf("Auditing %d manifest rows...\n", n);
for i = 1:n
    parts = split(M.relpath(i), "/");
    if M.root(i) == "pv"     % PlantVillage: <class folder>/<file>; folder spellings differ by punctuation only
        folderOk(i) = normaliseName(parts(1)) == pvFolders(M.y(i) + 1);
    else                     % PlantDoc: <train|test>/<class folder>/<file>
        folderOk(i) = numel(parts) >= 3 && parts(2) == pdFolders(M.y(i) + 1);
    end
    if ~folderOk(i), note(i) = addNote(note(i), "class folder does not match mapping"); end
    if ~exists(i)
        note(i) = addNote(note(i), "missing file");
        continue
    end

    fid = fopen(M.Path(i), "r");
    bytes = fread(fid, Inf, "*uint8");
    fclose(fid);
    digest = typecast(hasher.digest(typecast(bytes, "int8")), "uint8");
    md5ok(i) = string(sprintf("%02x", digest)) == M.md5(i);
    if ~md5ok(i), note(i) = addNote(note(i), "MD5 differs from manifest"); end

    try
        info = imfinfo(M.Path(i));
        format(i) = string(info(1).Format);
        colourType(i) = string(info(1).ColorType);
        imgWidth(i) = info(1).Width;
        imgHeight(i) = info(1).Height;
        if numel(info) > 1, note(i) = addNote(note(i), sprintf("%d frames, first used", numel(info))); end
    catch err
        note(i) = addNote(note(i), "imfinfo failed: " + err.message);
    end
    try
        preprocessImage(M.Path(i), "decode", Config=cfg);
        readable(i) = true;
    catch err
        note(i) = addNote(note(i), "decode failed: " + err.message);
    end
    if mod(i, 2000) == 0, fprintf("  %d / %d\n", i, n); end
end

rows = table(M.image_id, M.split, M.source, M.label, M.y, exists, md5ok, readable, folderOk, ...
    format, colourType, imgWidth, imgHeight, note, 'VariableNames', ["image_id", "split", "source", ...
    "label", "y", "exists", "md5_ok", "readable", "folder_ok", "format", "colour_type", "width", ...
    "height", "note"]);
writetable(rows, fullfile(outDir, "audit_rows.csv"));

% ---- class x partition counts
partitions = ["train", "val", "test", "external"];
counts = zeros(numel(classNames), numel(partitions));
for p = 1:numel(partitions)
    counts(:, p) = countcats(M.Y(M.split == partitions(p)));
end
countTable = array2table(counts, 'VariableNames', partitions, 'RowNames', classNames);
countTable.total = sum(counts, 2);
writetable(countTable, fullfile(outDir, "counts.csv"), 'WriteRowNames', true);

% ---- exact duplicates (by manifest MD5) and cross-dataset overlap
[~, ~, group] = unique(M.md5);
groupSize = accumarray(group, 1);
dupGroups = find(groupSize > 1);
crossPartition = 0; crossDataset = 0;
for g = dupGroups'
    members = group == g;
    crossPartition = crossPartition + (numel(unique(M.split(members))) > 1);
    crossDataset = crossDataset + (numel(unique(M.source(members))) > 1);
end

% ---- summary
summary.generated = string(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss Z", "TimeZone", "local"));
summary.matlab = string(version);
summary.manifest = cfg.manifest;
summary.rows = n;
for p = partitions
    summary.partitions.(p) = struct("found", nnz(M.split == p), "expected", cfg.expectedCounts.(p));
end
summary.partitionCountsMatch = all(arrayfun(@(p) summary.partitions.(p).found == ...
    summary.partitions.(p).expected, partitions));
summary.classes = classNames;
summary.missingFiles = nnz(~exists);
summary.md5Mismatches = nnz(exists & ~md5ok);
summary.unreadable = nnz(exists & ~readable);
summary.classFolderMismatches = nnz(~folderOk);
summary.exactDuplicateGroups = numel(dupGroups);
summary.duplicateGroupsAcrossPartitions = crossPartition;
summary.duplicateGroupsAcrossDatasets = crossDataset;
for s = ["plantvillage", "plantdoc"]
    k = M.source == s & exists;
    shortSide = min(imgWidth(k), imgHeight(k));
    summary.images.(s) = struct( ...
        "formats", tally(format(k)), "colourTypes", tally(colourType(k)), ...
        "shortSideMin", min(shortSide), "shortSideMedian", median(shortSide, "omitnan"), ...
        "shortSideMax", max(shortSide), ...
        "upscaledToInput", nnz(shortSide < cfg.resizeShort), ...   % enlarged by the resize step
        "shortSideBelow100", nnz(shortSide < 100));
end
summary.problemRows = nnz(note ~= "");

fid = fopen(fullfile(outDir, "audit_summary.json"), "w");
fprintf(fid, "%s\n", jsonencode(summary, "PrettyPrint", true));
fclose(fid);
fprintf("Audit written to %s (%d problem rows)\n", outDir, summary.problemRows);
end

function s = normaliseName(s)
% 'Pepper__bell___Bacterial_spot' (Kaggle) and 'Pepper,_bell___Bacterial_spot' (GitHub) compare equal.
s = lower(regexprep(s, "[^A-Za-z0-9]", ""));
end

function note = addNote(note, text)
if note == "", note = text; else, note = note + "; " + text; end
end

function t = tally(values)
% Counts per distinct value, as a struct for JSON (field names made valid).
t = struct();
[u, ~, idx] = unique(values);
c = accumarray(idx, 1);
for k = 1:numel(u)
    t.(matlab.lang.makeValidName(u(k))) = c(k);
end
end
