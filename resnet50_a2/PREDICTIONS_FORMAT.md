# Shared predictions format (for comparing group models)

To compare models fairly (McNemar test, A1 Section 5.3), every model must be evaluated on the
**same external images** from the shared `outputs/manifest.csv` (rows with `split == "external"`),
and export one CSV with these columns:

| column | meaning |
|---|---|
| `image_id` | copied from the manifest, unchanged (e.g. `pd/test/Tomato leaf/abc.jpg`) |
| `y_true` | true class index (manifest column `y`, 0-11, order of `class_mapping.json`) |
| `y_pred` | predicted class index, same 0-11 order |

Optional: `p_0` ... `p_11` class probabilities.

Minimal export example (PyTorch / any framework):
```python
import pandas as pd
ext = pd.read_csv("outputs/manifest.csv").query("split == 'external'")
# ... run your model over ext, in this row order, to get `preds` (list of ints 0-11)
pd.DataFrame({"image_id": ext.image_id, "y_true": ext.y, "y_pred": preds}) \
  .to_csv("predictions_external.csv", index=False)
```
For Ultralytics YOLO11s-cls, map its class names back to the 0-11 indices of `class_mapping.json`
(YOLO sorts class folders alphabetically, which is a different order).

Do the same for the PlantVillage internal test (`split == "test"`) as `predictions_internal.csv`.
