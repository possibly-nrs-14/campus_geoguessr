#   IIIT-H Campus Region Prediction Project

For this project, the dataset was entirely made by IIIT-H students; each student took 55 pictures of the campus. These images were then used for 3 tasks:
- **RegionId prediction:** Predict the region id (campus was divided into 15 regions for the task so possible ids were 1-15) of an image
- **Latitude and Longitude Prediction:** Predict the latitude and longitude (scaled) of an image
- **Angle prediction:** Predict the angle (0-360) relative to absolute north (0 degrees)
### **1. Region ID classifier (`region.py`)**

1. **Backbone:** *ConvNeXt‑Large* – modern CNN/Transformer hybrid, **ImageNet‑1K pre‑trained**.
2. **Fine‑tuning:** last layer re‑initialised to 15 classes; backbone kept trainable but with 10× lower LR (differential learning‑rates).
3. **Augmentation:** 256 → random‑crop 224, horizontal‑flip, colour‑jitter; ImageNet mean/σ normalisation.
4. **Optimiser / LR:** Adam W + 5‑epoch linear warm‑up then cosine decay; weight‑decay 0.05.


---

### **2. Latitude / Longitude multi‑task (`latlong.py`)**

1. **Architecture:** *ConvNeXt‑Large* backbone shared by two heads

   * **cls‑head** (15‑way softmax) & **reg‑head** (2‑D linear).
2. **Training objective:** `CE + λ · MSE₍z₎`, with **λ = 15** so regression signal equals CE magnitude.
3. **Data cleaning:** training CSV clipped to a tight geo‑box (218 k–222 k, 140.5 k–146 k) – removes the extreme outliers visible on the scatter‑plot → model focuses on dense campus cluster.
4. **Validation filtering:** images whose numeric IDs {95,145,…} are mislabeled are dropped.
5. **Normalisation:** coordinates → z‑scores per (filtered) train set; prediction re‑scaled at inference.
6. **Augmentation:** same as Region‑ID but milder hue/brightness to keep geo cues stable.
7. **Scheduler:** 5‑epoch warm‑up + cosine; Adam W, WD 0.05.
8. **Output files:** `latlon_multitask.pt`

---

### **3. Camera‑angle regressor (`angle.py`)**

1. **Backbone:** *Swin‑Large* (Transformer) pre‑trained on ImageNet‑22K, features only.
2. **Head:** 2‑unit linear → predicts **(sin θ, cos θ)**; keeps angle periodicity.
3. **Loss:** **CosineEmbeddingLoss** between predicted and target unit vectors → directly minimises angular deviation.
4. **Learning‑rate policy:** Adam W 1e‑4, warm‑up 5, cosine anneal.
5. **Metric:** Mean‑Absolute‑Angular‑Error (MAAE) printed each epoch; best ckpt saved as `angle_regressor.pt`.


---

