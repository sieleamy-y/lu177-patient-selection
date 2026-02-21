# AI-Guided Patient Selection for Lu-177 PSMA Therapy

**Kaggle Google AI Hackathon - Health AI Developer Foundations**  
*Bridging quantitative dosimetry and clinical decision-making*

---

## Overview

AI-powered clinical decision support system for Lu-177 PSMA radioligand therapy patient selection using Gemini and Medgemma (Health AI Developer Foundations).

**Problem:** 40-60% of patients selected by SUV criteria alone don't respond to Lu-177 therapy, and 12% experience severe nephrotoxicity.

**Solution:** Open-source dosimetry pipeline + Gemini/medgemma AI clinical layer = evidence-based eligibility recommendations with structured guidance for dose adjustments and monitoring.

---

## Key Features

✅ **AI Clinical Interpretation** - Gemini/medgemma translates dosimetry into actionable recommendations  
✅ **EANM Guideline Integration** - Evidence-based eligibility assessment  
✅ **Risk Stratification** - Clear/borderline/high-risk patient identification  
✅ **Structured Output** - Eligibility, dose recommendations, monitoring protocols  
✅ **Open-Source Architecture** - No expensive proprietary software required

---

## Technical Architecture

### Dosimetry Pipeline (Pseudocode)
- **Z-Rad**: Vendor-neutral SUV extraction
- **TotalSegmentator**: AI-based organ segmentation
- **MIRD Formalism**: Absorbed dose calculations
- **OLINDA/EXM S-values**: Reference radiation transport data

### AI Clinical Layer (Implemented)
- **Gemini 2.5-flash** (HAI-DEF model)
- Interprets kidney doses in clinical context
- Applies EANM/ESSO treatment guidelines
- Generates structured clinical recommendations

---

## Installation
```bash
# Clone repository
git clone https://github.com/sieleamy-y/lu177-patient-selection.git
cd lu177-patient-selection

# Create conda environment
conda create -n medgemma python=3.11
conda activate medgemma

# Install dependencies
pip install -r requirements.txt

# Set API key
export GOOGLE_API_KEY=your_key_here  # Mac/Linux
set GOOGLE_API_KEY=your_key_here     # Windows
```

---

## Quick Start

### Run Demo Notebook
```bash
jupyter notebook
# Open: AI-Guided Lu-177 Patient Selection Demo.ipynb
```

### Use Clinical Assistant
```python
from gemini_helper import ClinicalAssistant

assistant = ClinicalAssistant()

recommendation = assistant.check_eligibility(
    kidney_dose=12.5,
    egfr=45,
    predicted_lu_dose=18.3,
    kidney_history="Stage 3a CKD"
)

print(recommendation)
```

---

## Demo Cases

### Case 1: Clear Eligibility ✅
- Kidney dose: 10 Gy, eGFR: 70 mL/min
- **Output:** Standard protocol (7.4 GBq × 4-6 cycles)

### Case 2: Borderline ⚠️
- Kidney dose: 12.5 Gy, eGFR: 45 mL/min, Stage 3a CKD
- **Output:** Eligible with modifications (5.5 GBq, extended cycles, enhanced monitoring)

### Case 3: High Risk 🚫
- Kidney dose: 20 Gy, eGFR: 35 mL/min, Stage 3b CKD
- **Output:** High nephrotoxicity risk, major dose reduction or alternatives required

---

## Files

- `gemini_helper.py` - Gemini AI clinical assistant implementation
- `dosimetry_pseudocode.py` - Technical documentation of dosimetry workflow
- `AI-Guided Lu-177 Patient Selection Demo.ipynb` - Interactive demo notebook
- `requirements.txt` - Python dependencies

---

## Clinical Impact

**For Resource-Limited Settings:**
- Enables safer Lu-177 therapy implementation
- Evidence-based patient selection without expensive software
- Prevents inappropriate patient exclusions
- Identifies high-risk patients requiring modifications

**First in Sub-Saharan Africa:**
- Kenya's first open-source dosimetry framework
- Addresses healthcare infrastructure challenges
- Scalable to other theranostic applications

---

## Future Development

**Short-term:**
- Process real DICOM data from Kenyan patients
- Validate against OLINDA/EXM gold-standard
- Voxel-level dose heterogeneity analysis (DPK)

**Medium-term:**
- Multi-center deployment across East Africa
- Local MedGemma deployment for offline capability
- PACS integration

**Long-term:**
- Extend to other theranostic pairs (DOTATATE, FAP inhibitors)
- Prospective clinical trial validation
- Regulatory pathway for clinical decision support

---

## References

1. Peters et al. (2022). Ga-68 PSMA PET as predictor for Lu-177 doses. *Eur J Nucl Med Mol Imaging*
2. Sartor et al. (2021). Lu-177-PSMA-617 for mCRPC. *NEJM*
3. Sandgren et al. (2019). Ga-68 PSMA biodistribution. *Eur J Nucl Med*
4. Fritsak et al. (2025). Vendor-neutral SUV calculation. *arXiv:2410.13348*
5. Wasserthal et al. (2023). TotalSegmentator. *Radiology: AI*

---

## Author

**Amy Cheruto Siele**  
MSc Nuclear Medicine Student | Diagnostic Radiographer  
University of Nairobi | RFH Healthcare Kenya

**Competition:** Kaggle Google AI Hackathon - Health AI Developer Foundations  
**Date:** February 2026

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

This project leverages Google's Health AI Developer Foundations (HAI-DEF) and the open-source medical imaging community's tools (Z-Rad, TotalSegmentator) to democratize advanced healthcare technology.
## Live Demos

**MedGemma Implementation (Kaggle Notebook with GPU):**
https://www.kaggle.com/code/amysiele/medgemma-working-demo

**Local Gemini Implementation:**
See `AI-Guided Lu-177 Patient Selection Demo.ipynb` in this repository
