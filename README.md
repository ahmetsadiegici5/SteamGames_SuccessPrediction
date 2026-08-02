# Steam Games Pre-Launch Visibility Prediction
## Phase 3: Final Defense - ML Capstone Project

**Project Goal:** Predict which games will achieve high visibility (owners > median) on Steam using pre-launch features only.

---

## 📊 Project Structure

```
ML_FinalProject/
├── Phase3_Final_Presentation_4Models.pptx  ← FINAL PRESENTATION (15 min)
├── create_phase3_graphics.py               ← Graphics generator
├── make_phase3_pptx.py                     ← Presentation builder
├── steam.csv                               ← Raw dataset (27k+ games)
├── ML_FinalProject_Data/                   ← Backup data
└── presentation_assets/
    └── figures/                            ← All generated charts (10)
```

---

## 🎯 Key Results

### Model Performance (4 Models Compared)
| Model | Recall | Precision | F1 | Accuracy |
|-------|--------|-----------|-----|----------|
| **XGBoost (Winner)** | **64.8%** | **41.9%** | **0.509** | 62.7% |
| Random Forest | 56.9% | 39.8% | 0.467 | 60.5% |
| LR (Balanced) | 45.9% | 35.5% | 0.401 | 56.8% |
| Dummy Baseline | 0.0% | N/A | 0.000 | 70.2% |

### Business Impact
- **Hits Captured:** 1,030 / 1,590 (64.8% recall vs 0% baseline)
- **Cost Reduction:** $8.9M (55.8% savings)
- **FN vs FP Trade-off:** 560 missed hits vs 921 false positives (acceptable ratio)

---

## 📈 Features Used (11 Pre-Launch Only)

### Strategic (High Impact)
- **has_publisher** (0.118 importance): Marketing backing proxy
- **has_multiplayer** (0.106): Community potential
- **english** (0.057): Global market access

### Meta (Distribution)
- **platform_count**: Cross-platform presence
- **name_length**: Professionalism proxy (0.133)

### Content
- **is_free** / **is_mature**: Pricing & rating strategy

### Timing (Low Impact)
- **season_Spring/Summer/Fall/Winter** (<2pp): Negligible effect

**❌ Excluded for Leakage:**
- dev_experience (derived from past success → circular logic)
- tag_count (post-launch community data)

---

## 🔬 Methodology

### 1. Data Cleaning
- CSV robust parsing (27,042 → 26,672 rows)
- Owners range parsing: "10,000–20,000" → 15,000 (midpoint)
- Release date → datetime for seasonality extraction

### 2. Target Definition
- **Success = 1:** owners_midpoint > dataset median
- **Failure = 0:** owners_midpoint ≤ median
- Ratio: 29.4% success, 70.6% failure (imbalanced)

### 3. Feature Engineering
- Binary encoding (0/1) for all features
- Stratified 80/20 train-test split
- StandardScaler normalization

### 4. Modeling
- Baseline: Dummy classifier (most frequent)
- Linear: Logistic Regression with class_weight='balanced'
- Ensemble: Random Forest (100 trees, balanced)
- Boosting: XGBoost (scale_pos_weight tuned to FN > FP cost)

### 5. Evaluation
- **Primary Metric:** Recall (capture hits)
- **Secondary:** Precision (budget discipline), F1 (balance)
- **Validation:** 5-fold stratified cross-validation
- **Risk View:** Confusion matrices with cost framing

---

## 🚀 How to Regenerate

### Step 1: Generate Graphics
```bash
python create_phase3_graphics.py
```
Outputs: 10 PNG charts to `presentation_assets/figures/`

### Step 2: Build Presentation
```bash
python make_phase3_pptx.py
```
Outputs: `Phase3_Final_Presentation_4Models.pptx`

**Requirements:**
- pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, python-pptx

---

## 🧭 Minimal Dashboard (Streamlit)

Phase 3 gerekliliğindeki “Engine + Dashboard” kısmını göstermek için tek sayfalık, minimal bir Streamlit arayüzü eklenmiştir.

**Ne gösteriyor?**
- Model seçimi (XGBoost varsa), threshold slider ile karar eşiği
- Metrics: Accuracy / Precision / Recall / F1
- Confusion Matrix + Precision–Recall Curve
- Feature importance/coef özeti

**Çalıştırma**
```bash
pip install streamlit
streamlit run streamlit_app.py
```

Not: XGBoost dashboard’da opsiyoneldir. Yüklü değilse Logistic Regression / Random Forest ile çalışır.

---

## 💡 Actionable Insights

### What Drives Success
1. **Secure Publisher Backing** (+14pp) - Biggest lever
2. **Build Multiplayer/Co-op** (+12.8pp) - Community potential
3. **Platform Diversification** (11.1%) - Wider reach
4. **English Localization** (+8.5pp) - Global audience
5. **Avoid Free-to-Play** at launch (-6.3pp) - Perceived value

### What Doesn't Matter
- Release timing (seasonal variance < 2pp)
- Complex feature engineering (simple binary features work)

---

## 📋 Files

| File | Purpose |
|------|---------|
| `create_phase3_graphics.py` | Trains 4 models, generates 10 charts |
| `make_phase3_pptx.py` | Assembles 10-slide presentation deck |
| `steam.csv` | 27,042 Steam games dataset |
| `Phase3_Final_Presentation_4Models.pptx` | **FINAL DELIVERABLE** |
| `presentation_assets/figures/` | All generated visualizations |

---

## ⚠️ Integrity Notes

### No Leakage
✅ All features are **pre-launch only** (available before release)
✅ No post-launch signals (reviews, playtime, actual sales)
✅ No circular reasoning (dev history excluded)

### Model Code
✅ All Python code is original and fully documented
✅ Standard sklearn/XGBoost libraries (no copy-paste solutions)
✅ Each function has clear business logic

### Reproducibility
✅ Fixed random_state=42 for all models
✅ Stratified splits preserve class balance
✅ Scaling applied consistently train→test

---

## 📅 Timeline

| Phase | Deliverable | Status |
|-------|-------------|--------|
| Phase 1 | 5-min Pitch | ✅ Completed |
| Phase 2 | 7-10 min Progress Report | ✅ Completed |
| **Phase 3** | **15-min Final Defense + Code** | **🔴 IN PROGRESS** |

---

## 👤 Author

**Student:** Ahmet Sadi Egici  
**Course:** Machine Learning Capstone Project  
**Date:** January 2026  
**Dataset:** Steam Store (Kaggle)

---

## 🔗 References

- **Primary Metric:** Recall optimization for asymmetric costs
- **Class Imbalance:** Handled via class_weight + threshold tuning
- **Feature Extraction:** Domain knowledge (game industry signals)
- **Validation:** Stratified k-fold cross-validation (k=5)

---

**Status:** ✅ Ready for Phase 3 Final Defense
