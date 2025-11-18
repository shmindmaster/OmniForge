## 1. Objective

Execute the nail analysis pipeline on a designated test image set to:

* Generate **segmentation masks** for all visible nails.
* Compute **metric measurements** for each detected nail.
* Produce **visual overlays** to validate segmentation accuracy.

This run will serve as a **validation of model performance** and **measurement algorithm precision.**

---

## 2. Input Data

Process all images located in the project’s `handsy_in/` directory (also attached for reference):

```
handsy_in\V01_straight_clean.jpg
handsy_in\V02_noisy_extra.jpg
handsy_in\V03_straight_calibrator.jpg
handsy_in\V09AI_straight_clean_darker.jpg
handsy_in\V09AI_straight_clean.jpg
```

---

## 3. Technical Requirements

**Segmentation**

* For each input image, generate a **binary PNG mask** of all detected nails.
* Pixel values: background = `0` (black), nail = `255` (white).

**Measurement**

* For each detected nail, compute the following metrics (in **millimeters**):

  * `length_mm`: Length along the nail’s principal axis.
  * `width_prox_mm`: Width at 25% of the nail’s length (proximal).
  * `width_mid_mm`: Width at 50% of the nail’s length (midpoint).
  * `width_dist_mm`: Width at 75% of the nail’s length (distal).

**Scaling**

* Derive **pixel-to-mm conversion** from in-image scale references.
* Prioritize the **20 mm ArUco marker** for highest accuracy.
* Document the **scaling method used per image** in the output.

**Visualization**

* For each image, generate an **overlay image** (PNG or JPG) showing:

  * The original image with segmentation **contours drawn over detected nails**.

---

## 4. Deliverables

All outputs must be saved under the `handsy_out/` directory:

1. **Masks (`/masks`)**

   * Binary mask images (PNG) — one per input file.
   * Example: `V01_straight_clean_mask.png`

2. **Overlays (`/overlays`)**

   * Visualization images (PNG or JPG) with segmentation contours.
   * Example: `V01_straight_clean_overlay.png`

3. **Measurements (`measurements.csv`)**

   * One CSV file at the root.
   * Each row = one nail.
   * Required columns:
     `image`, `nail_index`, `length_mm`, `width_prox_mm`, `width_mid_mm`, `width_dist_mm`, `scale_method`

4. **Methods Note (`methods_note.md`)**

   * A brief (≤1 page) technical summary including:

     * Model architecture used (e.g., YOLOv11m-seg)
     * Preprocessing steps
     * Pixel-to-mm scaling method
     * Approximate runtime (e.g., ms per image)

---

## 5. Pre-Execution Checklist

Before running the pipeline, confirm:

* [ ] Model weights are loaded and compatible with the current environment
* [ ] Python environment and dependencies are fully installed and tested
* [ ] GPU acceleration (if applicable) is available and configured
* [ ] ArUco marker detection module is functioning
* [ ] Output directories are clean and write-accessible

---


