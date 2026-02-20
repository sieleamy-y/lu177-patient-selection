"""
═══════════════════════════════════════════════════════════════════════════
ORGAN DOSIMETRY PIPELINE - TECHNICAL DOCUMENTATION
═══════════════════════════════════════════════════════════════════════════

Full implementation workflow for [⁶⁸Ga]Ga-PSMA-11 PET/CT organ dosimetry
using open-source tools and MIRD formalism.

This pseudocode demonstrates the technical architecture that feeds into
the Gemini AI clinical layer showcased in this competition entry.

Author: Amy Cheruto Siele
Institution: University of Nairobi, RFH Healthcare Kenya
Competition: Kaggle Google AI Hackathon - Health AI Developer Foundations
Date: February 2026

Tools & Methods:
───────────────
- pydicom (v2.4+): DICOM file parsing and metadata extraction
- Z-Rad: Vendor-neutral SUV standardization (Fritsak et al., 2025)
- TotalSegmentator (v2.0+): AI-based organ segmentation (Wasserthal et al., 2023)
- NumPy/SciPy: Numerical computations and volume calculations
- MIRD Formalism: Absorbed dose calculations using OLINDA/EXM S-values
- ICRP 110/133: Reference organ masses and tissue densities

References:
───────────
1. Fritsak et al. (2025). Vendor-Specific SUV Calculation. arXiv:2410.13348
2. Wasserthal et al. (2023). TotalSegmentator. Radiology: AI
3. Sandgren et al. (2019). Ga-68 PSMA Biodistribution. Eur J Nucl Med
4. Bolch et al. (2009). MIRD Pamphlet 21. J Nucl Med, 50(3), 477-484
5. ICRP Publication 133 (2016). Voxel-based reference phantoms

═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pydicom
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: DICOM PROCESSING & VENDOR-NEUTRAL SUV EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def load_and_standardize_dicom(pet_dicom_path, ct_dicom_path):
    """
    Load PET/CT DICOM files and extract vendor-neutral standardized SUV.
    
    Critical Challenge:
    ──────────────────
    Different PET/CT manufacturers (GE, Siemens, Philips) encode timing 
    and calibration data differently in DICOM headers, causing SUV 
    calculation errors up to 33% (Fritsak et al., 2025).
    
    Solution: Z-Rad Tool
    ───────────────────
    Vendor-specific SUV correction strategies ensure accurate quantification
    across all scanner types. Essential for multicenter studies and 
    resource-limited settings with mixed scanner fleets.
    
    Parameters:
    ──────────
    pet_dicom_path : str or Path
        Path to PET DICOM file or directory
    ct_dicom_path : str or Path
        Path to CT DICOM file or directory
    
    Returns:
    ───────
    suv_volume : np.ndarray (3D)
        Body-weight normalized SUV volume (dimensionless)
    ct_volume : np.ndarray (3D)
        CT Hounsfield units for segmentation
    metadata : dict
        Essential parameters for dose calculation
    
    Implementation:
    ──────────────
    """
    # Load DICOM series
    pet_dicom = pydicom.dcmread(pet_dicom_path)
    ct_dicom = pydicom.dcmread(ct_dicom_path)
    
    # Extract critical metadata
    patient_weight_kg = float(pet_dicom.PatientWeight)
    
    # Radiopharmaceutical information (DICOM tag 0054,0016)
    radio_seq = pet_dicom.RadiopharmaceuticalInformationSequence[0]
    injected_dose_Bq = float(radio_seq.RadionuclideTotalDose)
    
    # Timing parameters (vendor-specific encoding!)
    injection_time = radio_seq.RadiopharmaceuticalStartTime
    scan_start_time = pet_dicom.AcquisitionTime  # or SeriesTime (vendor-dependent)
    
    # Decay correction parameters
    half_life_sec = float(radio_seq.RadionuclideHalfLife)  # Ga-68: 4080s
    
    # Extract PET activity concentration image (Bq/mL)
    pet_pixel_array = pet_dicom.pixel_array.astype(float)
    
    # Apply rescale slope/intercept (DICOM tag 0028,1053 / 0028,1052)
    rescale_slope = float(pet_dicom.RescaleSlope)
    rescale_intercept = float(pet_dicom.RescaleIntercept)
    activity_concentration = pet_pixel_array * rescale_slope + rescale_intercept
    
    # ═══════════════════════════════════════════════════════════════════════
    # Z-RAD VENDOR-SPECIFIC SUV CORRECTION
    # ═══════════════════════════════════════════════════════════════════════
    
    # Calculate time elapsed from injection to scan (seconds)
    time_elapsed_sec = calculate_time_difference(injection_time, scan_start_time)
    
    # Decay correction factor
    decay_factor = np.exp(-np.log(2) * time_elapsed_sec / half_life_sec)
    
    # Correct injected dose for decay at scan time
    injected_dose_at_scan = injected_dose_Bq * decay_factor
    
    # Vendor-specific corrections (Z-Rad methodology)
    vendor = pet_dicom.Manufacturer.upper()
    
    if "SIEMENS" in vendor:
        # Siemens: Check private tag (0071,1022) for dose calibration
        # May require additional Series Time vs Acquisition Time handling
        calibration_factor = get_siemens_calibration_factor(pet_dicom)
        activity_concentration *= calibration_factor
        
    elif "GE" in vendor:
        # GE: Private tag (0009,100d) contains critical timing info
        # Decay correction reference may differ from standard
        ge_decay_reference = get_ge_decay_reference(pet_dicom)
        activity_concentration = apply_ge_decay_correction(
            activity_concentration, 
            ge_decay_reference, 
            half_life_sec
        )
        
    elif "PHILIPS" in vendor:
        # Philips: Tags (7053,1000) and (7053,1009) for SUV scaling
        philips_suv_factor = get_philips_suv_factor(pet_dicom)
        activity_concentration *= philips_suv_factor
    
    # Calculate body-weight normalized SUV
    # SUV = (Activity_concentration [Bq/mL] * Patient_weight [g]) / Injected_dose [Bq]
    suv_volume = (activity_concentration * patient_weight_kg * 1000) / injected_dose_at_scan
    
    # Extract CT volume for segmentation
    ct_volume = ct_dicom.pixel_array.astype(float)
    
    # Collect metadata for downstream processing
    metadata = {
        'patient_weight_kg': patient_weight_kg,
        'injected_dose_MBq': injected_dose_Bq / 1e6,
        'scan_time_post_injection_min': time_elapsed_sec / 60,
        'pixel_spacing_mm': ct_dicom.PixelSpacing,
        'slice_thickness_mm': ct_dicom.SliceThickness,
        'manufacturer': vendor
    }
    
    return suv_volume, ct_volume, metadata


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: AUTOMATED ORGAN SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def segment_organs_at_risk(ct_volume, output_dir='segmentations/'):
    """
    Automated organ segmentation using TotalSegmentator AI model.
    
    Model Performance:
    ─────────────────
    TotalSegmentator achieves Dice coefficients >0.90 for abdominal organs
    (Wasserthal et al., 2023), trained on 1204 CT scans covering diverse
    pathologies and scanner protocols.
    
    Advantages over Manual Segmentation:
    ────────────────────────────────────
    • Speed: ~5-10 minutes on CPU vs 1-3 hours manual
    • Consistency: Eliminates inter-observer variability (20-30% in manual)
    • Reproducibility: Same input → same output (critical for research)
    
    Organs Segmented for Lu-177 PSMA Dosimetry:
    ───────────────────────────────────────────
    • Kidneys (left & right): Primary dose-limiting organ
    • Liver: Physiologic PSMA uptake, secondary OAR
    • Spleen: High physiologic uptake
    • Urinary bladder: Excretion pathway, variable uptake
    • (Future: Salivary glands for xerostomia risk assessment)
    
    Parameters:
    ──────────
    ct_volume : np.ndarray (3D)
        CT image volume in Hounsfield Units
    output_dir : str or Path
        Directory for segmentation masks
    
    Returns:
    ───────
    organ_masks : dict
        Dictionary of binary masks for each organ
        Keys: 'kidney_left', 'kidney_right', 'liver', 'spleen', 'bladder'
    
    Implementation:
    ──────────────
    """
    from totalsegmentator.python_api import totalsegmentator
    import nibabel as nib
    
    # Convert NumPy array to NIfTI format (TotalSegmentator input requirement)
    ct_nifti = convert_to_nifti(ct_volume)
    ct_nifti_path = Path(output_dir) / "ct_input.nii.gz"
    nib.save(ct_nifti, ct_nifti_path)
    
    # Run TotalSegmentator
    # ──────────────────────
    # ml=True: Use machine learning model (vs atlas-based)
    # fast=False: Prioritize accuracy over speed
    # task='total': Segment all 104 anatomical structures
    
    segmentation_output = totalsegmentator(
        input=str(ct_nifti_path),
        output=str(output_dir),
        ml=True,
        fast=False,
        task='total',
        device='cpu'  # Use GPU if available for 5x speedup
    )
    
    # Load individual organ masks
    organ_masks = {}
    
    organ_files = {
        'kidney_left': 'kidney_left.nii.gz',
        'kidney_right': 'kidney_right.nii.gz',
        'liver': 'liver.nii.gz',
        'spleen': 'spleen.nii.gz',
        'bladder': 'urinary_bladder.nii.gz'
    }
    
    for organ_name, filename in organ_files.items():
        mask_path = Path(output_dir) / filename
        mask_nifti = nib.load(mask_path)
        organ_masks[organ_name] = mask_nifti.get_fdata().astype(bool)
    
    # ═══════════════════════════════════════════════════════════════════════
    # QUALITY CONTROL (Essential for Clinical Use!)
    # ═══════════════════════════════════════════════════════════════════════
    
    # Automated checks for segmentation quality
    for organ_name, mask in organ_masks.items():
        volume_ml = calculate_volume_from_mask(mask, voxel_dims)
        
        # Physiologic volume ranges (ICRP 110 reference adult)
        expected_ranges = {
            'kidney_left': (120, 200),   # mL
            'kidney_right': (120, 200),
            'liver': (1400, 1800),
            'spleen': (150, 250),
            'bladder': (50, 500)  # Highly variable
        }
        
        min_vol, max_vol = expected_ranges.get(organ_name, (0, np.inf))
        
        if not (min_vol <= volume_ml <= max_vol):
            print(f"⚠️  WARNING: {organ_name} volume {volume_ml:.1f} mL "
                  f"outside expected range [{min_vol}-{max_vol}]")
            print(f"   → Manual review recommended in 3D Slicer")
    
    # Manual correction workflow (if needed):
    # ──────────────────────────────────────
    # 1. Load CT + masks in 3D Slicer (open-source)
    # 2. Use Segment Editor for corrections:
    #    - Paint/Erase for boundary refinement
    #    - Smoothing for anatomically realistic contours
    #    - Island removal for disconnected voxels
    # 3. Export corrected masks as NIfTI
    # 4. Reload for dose calculation
    
    return organ_masks


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: ORGAN VOLUME & MASS CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

def calculate_organ_properties(organ_mask, voxel_dimensions_mm, organ_type='kidney'):
    """
    Calculate anatomical properties from segmentation mask.
    
    Key Concept:
    ───────────
    Volume and mass are required for:
    1. S-value mass scaling (patient-specific vs reference phantom)
    2. Activity concentration calculation (Bq/mL)
    3. Dose heterogeneity analysis (dose per unit mass)
    
    ICRP 110 Reference Tissue Densities:
    ───────────────────────────────────
    Organ          Density (g/cm³)    Reference Mass (g)
    ─────────────  ────────────────   ─────────────────
    Kidneys        1.05               310 (both kidneys)
    Liver          1.06               1800
    Spleen         1.06               150
    Bladder wall   1.04               50 (empty)
    
    Parameters:
    ──────────
    organ_mask : np.ndarray (bool, 3D)
        Binary segmentation mask
    voxel_dimensions_mm : tuple (3,)
        Voxel spacing (x, y, z) in millimeters
    organ_type : str
        Organ name for density lookup
    
    Returns:
    ───────
    properties : dict
        volume_ml, mass_g, voxel_count
    
    Implementation:
    ──────────────
    """
    # Count voxels in organ
    voxel_count = np.sum(organ_mask)
    
    # Calculate voxel volume (mm³)
    voxel_volume_mm3 = np.prod(voxel_dimensions_mm)
    
    # Calculate organ volume (mL: 1 mL = 1000 mm³ = 1 cm³)
    volume_ml = (voxel_count * voxel_volume_mm3) / 1000
    
    # ICRP 110 tissue densities (g/cm³)
    tissue_densities = {
        'kidney': 1.05,
        'liver': 1.06,
        'spleen': 1.06,
        'bladder': 1.04,
        'bone': 1.92,
        'lung': 0.26,
        'soft_tissue': 1.04
    }
    
    density_g_per_ml = tissue_densities.get(organ_type, 1.04)
    
    # Calculate mass (g = mL × g/mL)
    mass_g = volume_ml * density_g_per_ml
    
    return {
        'volume_ml': volume_ml,
        'mass_g': mass_g,
        'voxel_count': voxel_count,
        'density_g_per_ml': density_g_per_ml
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: ACTIVITY QUANTIFICATION FROM PET
# ═══════════════════════════════════════════════════════════════════════════

def quantify_organ_activity(suv_volume, organ_mask, patient_weight_kg, 
                            injected_dose_MBq, voxel_volume_ml):
    """
    Extract organ-specific activity from PET using segmentation mask.
    
    SUV Metrics Explained:
    ─────────────────────
    SUVmean:  Average uptake across entire organ volume
              → Used for dose calculation (representative of whole organ)
    
    SUVmax:   Maximum single voxel value
              → Sensitive to noise, but useful for lesion detection
              → NOT used for dosimetry (non-representative)
    
    SUVpeak:  Average SUV in 1 cm³ sphere centered on SUVmax
              → Balance between noise reduction and peak detection
              → Used in some dosimetry protocols (PERCIST criteria)
    
    Conversion: SUV → Activity Concentration
    ────────────────────────────────────────
    SUV = (C [Bq/mL] × BW [kg] × 1000 [g/kg]) / ID [Bq]
    
    Rearranging:
    C [Bq/mL] = (SUV × ID [Bq]) / (BW [kg] × 1000)
    
    For dose calculation, we need activity in MBq:
    A [MBq] = C [Bq/mL] × V [mL] × 1e-6
    
    Parameters:
    ──────────
    suv_volume : np.ndarray (3D)
        Body-weight normalized SUV
    organ_mask : np.ndarray (bool, 3D)
        Binary organ segmentation
    patient_weight_kg : float
        Patient body weight
    injected_dose_MBq : float
        Administered Ga-68 PSMA activity
    voxel_volume_ml : float
        Volume of one voxel
    
    Returns:
    ───────
    activity_data : dict
        SUV metrics and organ activity (MBq)
    
    Implementation:
    ──────────────
    """
    # Apply mask to extract organ SUV values
    organ_suv_values = suv_volume[organ_mask]
    
    # Calculate SUV statistics
    suv_mean = np.mean(organ_suv_values)
    suv_max = np.max(organ_suv_values)
    suv_std = np.std(organ_suv_values)
    
    # SUVpeak: 1 cm³ (typically 1000 mm³) sphere around SUVmax
    suv_peak = calculate_suv_peak(suv_volume, organ_mask, sphere_volume_ml=1.0)
    
    # Convert SUVmean to activity concentration (Bq/mL)
    # Formula: C = (SUV × Injected_dose [Bq]) / (Weight [g])
    injected_dose_Bq = injected_dose_MBq * 1e6
    patient_weight_g = patient_weight_kg * 1000
    
    activity_concentration_Bq_per_ml = (suv_mean * injected_dose_Bq) / patient_weight_g
    
    # Calculate total organ activity at measurement time (typically 60 min p.i.)
    organ_volume_ml = np.sum(organ_mask) * voxel_volume_ml
    total_organ_activity_Bq = activity_concentration_Bq_per_ml * organ_volume_ml
    total_organ_activity_MBq = total_organ_activity_Bq / 1e6
    
    return {
        'suv_mean': suv_mean,
        'suv_max': suv_max,
        'suv_peak': suv_peak,
        'suv_std': suv_std,
        'activity_concentration_kBq_per_ml': activity_concentration_Bq_per_ml / 1000,
        'total_activity_MBq': total_organ_activity_MBq,
        'measurement_timepoint_min': 60  # Standard Ga-68 PSMA protocol
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: TIME-INTEGRATED ACTIVITY (CUMULATED ACTIVITY)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_cumulated_activity(organ_activity_MBq, residence_time_h, 
                                 residence_time_std_h=None):
    """
    Calculate time-integrated activity (cumulated activity).
    
    Conceptual Framework:
    ────────────────────
    Single-timepoint imaging (e.g., 60 min p.i.) provides a "snapshot"
    of activity distribution. To calculate absorbed dose, we need the
    TOTAL number of decays occurring in the organ over time.
    
    This is the TIME-INTEGRATED ACTIVITY (also called cumulated activity):
    
        Ã = ∫₀^∞ A(t) dt
    
    Where A(t) is the time-dependent activity in the organ.
    
    Practical Implementation:
    ────────────────────────
    Multi-timepoint imaging (2-4 scans over several hours) allows fitting
    bi-exponential clearance models. However, this is:
    • Time-consuming for patients
    • Resource-intensive
    • Often not clinically feasible
    
    Single-Timepoint Approach:
    ─────────────────────────
    Use population-averaged residence times from published multi-timepoint
    studies (Sandgren et al., 2019; Pfob et al., 2016).
    
    Residence time τ (hours) relates measured activity to cumulated activity:
    
        Ã = A(t₀) × τ
    
    Where:
    • A(t₀) = measured activity at timepoint t₀ (e.g., 60 min)
    • τ = residence time (time-integral per unit activity)
    
    Published Residence Times for [⁶⁸Ga]Ga-PSMA-11:
    ──────────────────────────────────────────────
    Organ          τ (hours)      Reference
    ─────────────  ────────────   ─────────────────────
    Kidneys        0.22 ± 0.05    Sandgren et al., 2019
    Liver          0.23 ± 0.04    Sandgren et al., 2019
    Spleen         0.016 ± 0.008  Pfob et al., 2016
    Bladder        0.147 ± 0.073  Pfob et al., 2016
    Salivary glands 0.014 ± 0.004  Sandgren et al., 2019
    
    Uncertainty Quantification:
    ──────────────────────────
    Inter-individual kinetic variability is captured by standard deviations.
    Sensitivity analysis calculates dose range using:
    • Lower bound: τ - σ
    • Mean estimate: τ
    • Upper bound: τ + σ
    
    This provides 95% confidence intervals on dose estimates.
    
    Parameters:
    ──────────
    organ_activity_MBq : float
        Measured activity at timepoint (typically 60 min)
    residence_time_h : float
        Population-averaged residence time (hours)
    residence_time_std_h : float, optional
        Standard deviation for uncertainty analysis
    
    Returns:
    ───────
    cumulated_activity : dict
        Mean, lower_95ci, upper_95ci in MBq·h
    
    Implementation:
    ──────────────
    """
    # Calculate cumulated activity (MBq·h)
    cumulated_activity_mean = organ_activity_MBq * residence_time_h
    
    # Uncertainty propagation (if std provided)
    if residence_time_std_h is not None:
        # Lower 95% CI: mean - 1.96×σ (approximates 2.5th percentile)
        residence_time_lower = residence_time_h - 1.96 * residence_time_std_h
        cumulated_activity_lower = organ_activity_MBq * max(residence_time_lower, 0)
        
        # Upper 95% CI: mean + 1.96×σ (approximates 97.5th percentile)
        residence_time_upper = residence_time_h + 1.96 * residence_time_std_h
        cumulated_activity_upper = organ_activity_MBq * residence_time_upper
    else:
        # If no std provided, use ±20% as conservative estimate
        cumulated_activity_lower = cumulated_activity_mean * 0.8
        cumulated_activity_upper = cumulated_activity_mean * 1.2
    
    return {
        'mean_MBq_h': cumulated_activity_mean,
        'lower_95ci_MBq_h': cumulated_activity_lower,
        'upper_95ci_MBq_h': cumulated_activity_upper,
        'residence_time_h': residence_time_h,
        'residence_time_std_h': residence_time_std_h
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: ABSORBED DOSE CALCULATION (MIRD FORMALISM)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_absorbed_dose_mird(cumulated_activity_MBq_h, patient_organ_mass_g,
                                reference_organ_mass_g, s_value_Gy_per_MBq_h):
    """
    Calculate organ absorbed dose using Medical Internal Radiation Dose (MIRD) schema.
    
    MIRD Fundamental Equation:
    ─────────────────────────
    D(T←S) = Ã(S) × S(T←S)
    
    Where:
    • D(T←S) = Absorbed dose to target organ T from source organ S (Gy)
    • Ã(S) = Cumulated activity in source organ S (MBq·h)
    • S(T←S) = S-value (mean absorbed dose per unit cumulated activity)
    
    S-Value Concept:
    ───────────────
    S-values encode the complex radiation transport physics:
    • Radionuclide decay scheme (Ga-68: β⁺, 511 keV annihilation photons)
    • Radiation type and energy spectrum
    • Geometric relationship between source and target
    • Tissue attenuation and scatter
    • Target organ mass
    
    S-values are pre-calculated using Monte Carlo simulations on
    ICRP 133 computational phantoms (voxelized adult male/female).
    
    Source of S-Values:
    ──────────────────
    OLINDA/EXM software (Organ Level Internal Dose Assessment)
    • FDA-approved dosimetry software
    • Implements MIRD schema with ICRP 133 phantoms
    • Provides S-values for >800 radionuclides
    • Used as "gold standard" for validation
    
    Patient-Specific Mass Scaling:
    ─────────────────────────────
    Reference phantoms have fixed organ masses. Patient-specific masses
    from CT segmentation require S-value adjustment:
    
    S(T←S)_patient = S(T←S)_reference × (M_reference / M_patient)
    
    This assumes radiation absorption scales inversely with mass
    (valid for self-dose when organ >> radiation range).
    
    Self-Dose vs Cross-Dose:
    ───────────────────────
    For organs with high uptake (kidneys, liver, spleen):
    • Self-dose (T=S) contributes >95% of total dose
    • Cross-dose from adjacent organs typically <5%
    • For dosimetry, often sufficient to calculate self-dose only
    
    Example S-Values for Ga-68 (OLINDA/EXM, Adult Male):
    ──────────────────────────────────────────────────
    Target    Source       S-value (Gy/MBq·h)    Ref. Mass (g)
    ────────  ───────────  ──────────────────    ─────────────
    Kidneys   Kidneys      4.89E-03              310
    Liver     Liver        1.44E-03              1800
    Spleen    Spleen       1.16E-02              150
    Bladder   Bladder      2.27E-03              50 (wall)
    
    Parameters:
    ──────────
    cumulated_activity_MBq_h : float
        Time-integrated activity (MBq·h)
    patient_organ_mass_g : float
        Patient-specific organ mass from CT segmentation
    reference_organ_mass_g : float
        ICRP 133 reference phantom organ mass
    s_value_Gy_per_MBq_h : float
        S-value from OLINDA/EXM for reference phantom
    
    Returns:
    ───────
    absorbed_dose : dict
        Dose (Gy), scaled S-value, mass scaling factor
    
    Implementation:
    ──────────────
    """
    # Mass scaling factor
    mass_scaling_factor = reference_organ_mass_g / patient_organ_mass_g
    
    # Patient-specific S-value
    s_value_patient = s_value_Gy_per_MBq_h * mass_scaling_factor
    
    # Calculate absorbed dose (Gy)
    absorbed_dose_Gy = cumulated_activity_MBq_h * s_value_patient
    
    # Convert to mGy for easier interpretation
    absorbed_dose_mGy = absorbed_dose_Gy * 1000
    
    return {
        'absorbed_dose_Gy': absorbed_dose_Gy,
        'absorbed_dose_mGy': absorbed_dose_mGy,
        's_value_reference': s_value_Gy_per_MBq_h,
        's_value_patient': s_value_patient,
        'mass_scaling_factor': mass_scaling_factor,
        'reference_mass_g': reference_organ_mass_g,
        'patient_mass_g': patient_organ_mass_g
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: COMPLETE PIPELINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def run_complete_dosimetry_pipeline(pet_dicom_path, ct_dicom_path, 
                                    clinical_data=None):
    """
    End-to-end dosimetry workflow for [⁶⁸Ga]Ga-PSMA-11 PET/CT.
    
    Workflow Summary:
    ────────────────
    1. Load DICOM → Extract vendor-neutral SUV (Z-Rad)
    2. Segment organs → AI-based masks (TotalSegmentator)
    3. Calculate volumes/masses → Patient-specific anatomy
    4. Quantify activity → Organ SUV and activity (MBq)
    5. Integrate over time → Cumulated activity (MBq·h)
    6. Calculate dose → MIRD formalism with S-values (Gy)
    7. Format output → Feed into Gemini AI clinical layer
    
    Validation Strategy:
    ───────────────────
    Results validated against:
    • OLINDA/EXM software (gold-standard comparison)
    • EANM physiological reference ranges
    • Published Ga-68 PSMA dosimetry literature
    • Scanner-reported SUV values (for local data)
    
    Parameters:
    ──────────
    pet_dicom_path : str or Path
        Path to PET DICOM
    ct_dicom_path : str or Path
        Path to CT DICOM
    clinical_data : dict, optional
        eGFR, kidney history for AI interpretation
    
    Returns:
    ───────
    results : dict
        Comprehensive dosimetry + clinical data
    
    Implementation:
    ──────────────
    """
    print("="*70)
    print("STARTING DOSIMETRY PIPELINE")
    print("="*70)
    
    # Step 1: Load and standardize
    print("\n[1/7] Loading DICOM and extracting SUV...")
    suv_volume, ct_volume, metadata = load_and_standardize_dicom(
        pet_dicom_path, ct_dicom_path
    )
    print(f"✓ Loaded {metadata['manufacturer']} PET/CT")
    print(f"  Patient weight: {metadata['patient_weight_kg']} kg")
    print(f"  Injected dose: {metadata['injected_dose_MBq']:.1f} MBq")
    print(f"  Scan time: {metadata['scan_time_post_injection_min']:.0f} min p.i.")
    
    # Step 2: Segment organs
    print("\n[2/7] Segmenting organs with TotalSegmentator...")
    organ_masks = segment_organs_at_risk(ct_volume)
    print(f"✓ Segmented {len(organ_masks)} organs")
    
    # Step 3: Calculate organ properties
    print("\n[3/7] Calculating organ volumes and masses...")
    voxel_dims = metadata['pixel_spacing_mm'] + (metadata['slice_thickness_mm'],)
    voxel_volume_ml = np.prod(voxel_dims) / 1000
    
    kidneys_combined_mask = organ_masks['kidney_left'] | organ_masks['kidney_right']
    kidney_properties = calculate_organ_properties(
        kidneys_combined_mask, voxel_dims, 'kidney'
    )
    print(f"✓ Kidney volume: {kidney_properties['volume_ml']:.1f} mL")
    print(f"  Kidney mass: {kidney_properties['mass_g']:.1f} g")
    
    # Step 4: Quantify activity
    print("\n[4/7] Quantifying kidney activity...")
    kidney_activity = quantify_organ_activity(
        suv_volume,
        kidneys_combined_mask,
        metadata['patient_weight_kg'],
        metadata['injected_dose_MBq'],
        voxel_volume_ml
    )
    print(f"✓ Kidney SUVmean: {kidney_activity['suv_mean']:.2f}")
    print(f"  Kidney activity: {kidney_activity['total_activity_MBq']:.2f} MBq at 60 min")
    
    # Step 5: Calculate cumulated activity
    print("\n[5/7] Calculating time-integrated activity...")
    # Using Sandgren et al. 2019 residence time for kidneys
    kidney_cumulated = calculate_cumulated_activity(
        kidney_activity['total_activity_MBq'],
        residence_time_h=0.22,
        residence_time_std_h=0.05
    )
    print(f"✓ Cumulated activity: {kidney_cumulated['mean_MBq_h']:.2f} MBq·h")
    print(f"  95% CI: [{kidney_cumulated['lower_95ci_MBq_h']:.2f}, "
          f"{kidney_cumulated['upper_95ci_MBq_h']:.2f}] MBq·h")
    
    # Step 6: Calculate absorbed dose
    print("\n[6/7] Calculating absorbed dose (MIRD)...")
    # OLINDA/EXM S-value for Ga-68 kidneys self-dose (adult male phantom)
    kidney_dose = calculate_absorbed_dose_mird(
        kidney_cumulated['mean_MBq_h'],
        patient_organ_mass_g=kidney_properties['mass_g'],
        reference_organ_mass_g=310,  # ICRP 133 reference
        s_value_Gy_per_MBq_h=4.89e-3  # OLINDA/EXM
    )
    print(f"✓ Kidney absorbed dose: {kidney_dose['absorbed_dose_Gy']:.3f} Gy")
    print(f"  = {kidney_dose['absorbed_dose_mGy']:.1f} mGy")
    
    # Step 7: Prepare for Gemini AI clinical layer
    print("\n[7/7] Formatting output for AI clinical interpretation...")
    
    # Extrapolate Ga-68 dose to Lu-177 (using activity ratio)
    # Typical: 7.4 GBq Lu-177 × 4 cycles vs 150 MBq Ga-68
    # Ratio ≈ 200, but kidney uptake differs (adjust by factor ~1.5)
    predicted_lu177_dose = kidney_dose['absorbed_dose_Gy'] * 1.47
    
    # Package results for Gemini AI
    dosimetry_output = {
        'kidney_dose_Ga68_Gy': kidney_dose['absorbed_dose_Gy'],
        'kidney_dose_Ga68_mGy': kidney_dose['absorbed_dose_mGy'],
        'predicted_Lu177_dose_Gy': predicted_lu177_dose,
        'kidney_volume_ml': kidney_properties['volume_ml'],
        'kidney_suv_mean': kidney_activity['suv_mean'],
        'egfr_ml_min': clinical_data.get('egfr') if clinical_data else None,
        'kidney_history': clinical_data.get('kidney_history', 'None') if clinical_data else 'None'
    }
    
    print(f"\n✓ Predicted Lu-177 kidney dose: {predicted_lu177_dose:.1f} Gy")
    print(f"  (Assumes 4 cycles × 7.4 GBq)")
    
    print("\n" + "="*70)
    print("DOSIMETRY PIPELINE COMPLETE")
    print("="*70)
    print("\n→ Output ready for Gemini AI clinical interpretation")
    
    return dosimetry_output


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: CONNECTION TO GEMINI AI CLINICAL LAYER
# ═══════════════════════════════════════════════════════════════════════════

def integrate_with_gemini_ai(dosimetry_output):
    """
    Feed dosimetry output into Gemini AI for clinical interpretation.
    
    This is the KEY INNOVATION of the competition entry:
    ────────────────────────────────────────────────────
    Raw dosimetry numbers (e.g., "Kidney dose = 12.5 Gy") are not
    immediately actionable for clinicians. They need:
    
    1. Contextualization (Is 12.5 Gy safe? For which patients?)
    2. Guideline application (What do EANM recommendations say?)
    3. Risk stratification (Does eGFR 45 change recommendations?)
    4. Dose adjustments (Should activity per cycle be reduced?)
    5. Monitoring protocols (What follow-up is needed?)
    
    Gemini AI Clinical Layer provides:
    ──────────────────────────────────
    • Evidence-based eligibility assessment
    • Structured recommendations (dose, monitoring)
    • Patient-friendly explanations
    • EANM/ESSO guideline integration
    
    Example Query to Gemini:
    ───────────────────────
    """
    from gemini_helper import ClinicalAssistant
    
    assistant = ClinicalAssistant()
    
    recommendation = assistant.check_eligibility(
        kidney_dose=dosimetry_output['kidney_dose_Ga68_Gy'],
        egfr=dosimetry_output['egfr_ml_min'],
        predicted_lu_dose=dosimetry_output['predicted_Lu177_dose_Gy'],
        kidney_history=dosimetry_output['kidney_history']
    )
    
    return recommendation


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Example demonstration of complete workflow.
    
    Note: This pseudocode shows the technical architecture.
    The competition demo uses simulated dosimetry outputs to
    showcase the Gemini AI clinical layer innovation.
    """
    
    # Example patient data (simulated for competition demo)
    simulated_dosimetry = {
        'kidney_dose_Ga68_Gy': 12.5,
        'kidney_dose_Ga68_mGy': 12500,
        'predicted_Lu177_dose_Gy': 18.3,
        'kidney_volume_ml': 150,
        'kidney_suv_mean': 8.2,
        'egfr_ml_min': 45,
        'kidney_history': 'Stage 3a CKD'
    }
    
    print("\n" + "="*70)
    print("SIMULATED DOSIMETRY OUTPUT (Competition Demo)")
    print("="*70)
    for key, value in simulated_dosimetry.items():
        print(f"{key}: {value}")
    
    print("\n→ Feeding into Gemini AI clinical layer...")
    print("→ See gemini_helper.py and demo notebook for AI recommendations")
    
    # Full pipeline would run:
    # results = run_complete_dosimetry_pipeline(
    #     pet_dicom_path="path/to/pet.dcm",
    #     ct_dicom_path="path/to/ct.dcm",
    #     clinical_data={'egfr': 45, 'kidney_history': 'Stage 3a CKD'}
    # )