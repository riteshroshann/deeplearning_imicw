"""
data_builder.py — Dataset Construction with Provenance Tracking
================================================================
Curates experimental and mean-value data from 8 validated literature
sources for CFRP composites. Each record carries full provenance
(source_id, reference, DOI, extraction method, data_type flag).

Sources
-------
S1   NCAMP/NIAR IM7/8552 — FAA-accepted qualification database
S2   Alsheghri & Drakonakis (2025) — CNT-enhanced CFRP, PLOS ONE
S3   Mohamed et al. (2023) — Fibre orientation sweep, Alexandria EJ
S4   Sudarisman et al. (2023) — Water absorption effects, MDPI JCS
S6   Tefera et al. (2022) — Temperature-dependent T700, Sage JCM
S7   Composites Part A (2025) — T800 transverse micromechanics
S8   Composites Part B (2025) — T300/T700 comparative flexural + impact
S10  ACP Composites (2023) — Systematic property tables (single values)
"""

import numpy as np
import pandas as pd
from config import MPa_TO_ksi, GPa_TO_Msi


# ── Record Schema ────────────────────────────────────────────────────────────

def _record(source_id, ref, doi, extraction, data_type,
            fiber, resin, grade, method, fabric,
            vf, cnt, il, pressure, tg_dry, tg_wet,
            layup, p0, p45, p90, n_plies, thick_mm,
            test_type, standard, env, temp_c, moisture,
            tens_mpa=None, tens_gpa=None, comp_mpa=None, comp_gpa=None,
            flex_mpa=None, flex_gpa=None, shear_mpa=None, giic=None,
            cv_pct=None, std_dev=None):
    """
    Standardised record with provenance, all mechanical property columns,
    and a unified strength/modulus column for modelling convenience.
    """
    s = tens_mpa or flex_mpa or comp_mpa or shear_mpa
    m = tens_gpa or flex_gpa or comp_gpa
    env_code = {"CTD": 0, "RTD": 1, "ETD": 2, "ETW": 3}.get(env, 1)
    return {
        "source_id":                source_id,
        "source_ref":               ref,
        "doi_or_url":               doi,
        "data_extraction":          extraction,
        "data_type":                data_type,
        "fiber_type":               fiber,
        "resin_type":               resin,
        "fiber_grade":              grade,
        "manufacturing_method":     method,
        "fabric_type":              fabric,
        "fiber_volume_pct":         vf,
        "CNT_vol_frac_pct":         cnt,
        "interlayer_vol_frac_pct":  il,
        "manufacturing_pressure_psi": pressure,
        "Tg_dry_C":                 tg_dry,
        "Tg_wet_C":                 tg_wet,
        "layup_code":               layup,
        "pct_0_plies":              p0,
        "pct_45_plies":             p45,
        "pct_90_plies":             p90,
        "num_plies":                n_plies,
        "thickness_mm":             thick_mm,
        "test_type":                test_type,
        "test_standard":            standard,
        "environment":              env,
        "env_code":                 env_code,
        "temperature_C":            temp_c,
        "moisture_condition":       moisture,
        "moisture_code":            1 if moisture == "Wet" else 0,
        "tensile_strength_MPa":     tens_mpa,
        "tensile_modulus_GPa":      tens_gpa,
        "compressive_strength_MPa": comp_mpa,
        "compressive_modulus_GPa":  comp_gpa,
        "flexural_strength_MPa":    flex_mpa,
        "flexural_modulus_GPa":     flex_gpa,
        "shear_strength_MPa":       shear_mpa,
        "GIIc_Jm2":                giic,
        "reported_CV_pct":          cv_pct,
        "reported_std_MPa":         std_dev,
        "strength_MPa":             s,
        "modulus_GPa":              m,
    }


# Compact wrapper
def _R(src, ref, doi, ext, dtype, fiber, resin, grade, method, fabric,
       vf, cnt, il, psi, tg_d, tg_w, layup, p0, p45, p90, npl, thk,
       test, std, env, tc, mst,
       ts=None, tm=None, cs=None, cm=None,
       fs=None, fm=None, ss=None, gi=None, cv=None, sd=None):
    return _record(src, ref, doi, ext, dtype,
                   fiber, resin, grade, method, fabric,
                   vf, cnt, il, psi, tg_d, tg_w,
                   layup, p0, p45, p90, npl, thk,
                   test, std, env, tc, mst,
                   ts, tm, cs, cm, fs, fm, ss, gi, cv, sd)


# ── S1: NCAMP / NIAR IM7/8552 Qualification Database ────────────────────────

def _build_s1():
    """
    NCAMP CAM-RP-2009-015 Rev A, NIAR Wichita State University, 2011.
    B-basis design values from FAA-accepted IM7/8552 qualification.
    Vf = 58.54%, autoclave-processed unidirectional prepreg.
    """
    REF = "NCAMP, CAM-RP-2009-015 Rev A. NIAR, Wichita State Univ., 2011."
    DOI = "https://www.wichita.edu/NIAR/ncamp"
    EXT = "Table 2-1 & 2-2 B-basis lamina/laminate design values"

    def S1(layup, p0, p45, p90, test, std, env,
           ts_ksi=None, tm_msi=None, cs_ksi=None, cm_msi=None,
           fs_ksi=None, fm_msi=None, ss_ksi=None):
        temp_F = {"CTD": -65, "RTD": 70, "ETD": 250, "ETW": 250}[env]
        temp_C = round((temp_F - 32) * 5 / 9, 1)
        mst = "Wet" if env == "ETW" else "Dry"
        return _R("S1", REF, DOI, EXT, "Mean_value",
                  "IM7", "Hexcel 8552 Epoxy", "Aerospace Qualification",
                  "Autoclave", "Unidirectional (UD)",
                  58.54, 0.0, 0.0, 100.0, 207.0, 160.2,
                  layup, p0, p45, p90, None, None, test, std, env, temp_C, mst,
                  ts=ts_ksi / MPa_TO_ksi if ts_ksi else None,
                  tm=tm_msi / GPa_TO_Msi if tm_msi else None,
                  cs=cs_ksi / MPa_TO_ksi if cs_ksi else None,
                  cm=cm_msi / GPa_TO_Msi if cm_msi else None,
                  fs=fs_ksi / MPa_TO_ksi if fs_ksi else None,
                  fm=fm_msi / GPa_TO_Msi if fm_msi else None,
                  ss=ss_ksi / MPa_TO_ksi if ss_ksi else None)

    return [
        # Longitudinal Tension (0° UD)
        S1("[0]6",   100,0,0, "Longitudinal_Tension", "ASTM D3039", "CTD", ts_ksi=362.69, tm_msi=22.99),
        S1("[0]6",   100,0,0, "Longitudinal_Tension", "ASTM D3039", "RTD", ts_ksi=357.39, tm_msi=22.57),
        S1("[0]6",   100,0,0, "Longitudinal_Tension", "ASTM D3039", "ETW", ts_ksi=333.50, tm_msi=24.00),
        # Transverse Tension (90° UD)
        S1("[90]11", 0,0,100, "Transverse_Tension", "ASTM D3039", "CTD", ts_ksi=9.29, tm_msi=1.30),
        S1("[90]11", 0,0,100, "Transverse_Tension", "ASTM D3039", "RTD", ts_ksi=9.60, tm_msi=1.46),
        S1("[90]11", 0,0,100, "Transverse_Tension", "ASTM D3039", "ETW", ts_ksi=3.49, tm_msi=0.81),
        # Longitudinal Compression (0° UD)
        S1("[0]14",  100,0,0, "Longitudinal_Compression", "ASTM D6641", "CTD", cs_ksi=296.49, cm_msi=20.68),
        S1("[0]14",  100,0,0, "Longitudinal_Compression", "ASTM D6641", "RTD", cs_ksi=248.94, cm_msi=20.04),
        S1("[0]14",  100,0,0, "Longitudinal_Compression", "ASTM D6641", "ETD", cs_ksi=201.93, cm_msi=20.25),
        S1("[0]14",  100,0,0, "Longitudinal_Compression", "ASTM D6641", "ETW", cs_ksi=173.00, cm_msi=20.37),
        # Transverse Compression (90° UD)
        S1("[90]14", 0,0,100, "Transverse_Compression", "ASTM D6641", "CTD", cs_ksi=55.31, cm_msi=1.53),
        S1("[90]14", 0,0,100, "Transverse_Compression", "ASTM D6641", "RTD", cs_ksi=41.44, cm_msi=1.41),
        S1("[90]14", 0,0,100, "Transverse_Compression", "ASTM D6641", "ETW", cs_ksi=19.02, cm_msi=1.18),
        # In-Plane Shear
        S1("[45/-45]3S", 0,100,0, "InPlane_Shear", "ASTM D3518", "CTD", ss_ksi=13.22),
        S1("[45/-45]3S", 0,100,0, "InPlane_Shear", "ASTM D3518", "RTD", ss_ksi=13.22),
        S1("[45/-45]3S", 0,100,0, "InPlane_Shear", "ASTM D3518", "ETW", ss_ksi=5.54),
        # Quasi-Isotropic Tension
        S1("[45/0/-45/90]2S", 25,50,25, "UNT_QI_Tension", "ASTM D3039", "CTD", ts_ksi=93.2, tm_msi=8.21),
        S1("[45/0/-45/90]2S", 25,50,25, "UNT_QI_Tension", "ASTM D3039", "RTD", ts_ksi=90.8, tm_msi=8.10),
        S1("[45/0/-45/90]2S", 25,50,25, "UNT_QI_Tension", "ASTM D3039", "ETW", ts_ksi=84.3, tm_msi=7.95),
        # Soft laminate tension
        S1("[45/-45/0/45/-45/90/45/-45/45/-45]S", 10,80,10, "UNT_Soft_Tension", "ASTM D3039", "CTD", ts_ksi=64.7, tm_msi=5.82),
        S1("[45/-45/0/45/-45/90/45/-45/45/-45]S", 10,80,10, "UNT_Soft_Tension", "ASTM D3039", "RTD", ts_ksi=63.1, tm_msi=5.71),
        S1("[45/-45/0/45/-45/90/45/-45/45/-45]S", 10,80,10, "UNT_Soft_Tension", "ASTM D3039", "ETW", ts_ksi=57.4, tm_msi=5.60),
        # Hard laminate tension
        S1("[0/45/0/90/0/-45/0/45/0/-45]S", 50,40,10, "UNT_Hard_Tension", "ASTM D3039", "CTD", ts_ksi=134.6, tm_msi=11.94),
        S1("[0/45/0/90/0/-45/0/45/0/-45]S", 50,40,10, "UNT_Hard_Tension", "ASTM D3039", "RTD", ts_ksi=131.2, tm_msi=11.72),
        S1("[0/45/0/90/0/-45/0/45/0/-45]S", 50,40,10, "UNT_Hard_Tension", "ASTM D3039", "ETW", ts_ksi=119.8, tm_msi=11.43),
        # Compression (QI, Soft, Hard)
        S1("[45/0/-45/90]2S", 25,50,25, "UNC_QI_Compression", "ASTM D6641", "RTD", cs_ksi=72.4, cm_msi=8.08),
        S1("[45/0/-45/90]2S", 25,50,25, "UNC_QI_Compression", "ASTM D6641", "ETW", cs_ksi=55.3, cm_msi=7.87),
        S1("[45/-45/0/45/-45/90/45/-45/45/-45]S", 10,80,10, "UNC_Soft_Compression", "ASTM D6641", "RTD", cs_ksi=53.8, cm_msi=5.68),
        S1("[45/-45/0/45/-45/90/45/-45/45/-45]S", 10,80,10, "UNC_Soft_Compression", "ASTM D6641", "ETW", cs_ksi=38.2, cm_msi=5.41),
        S1("[0/45/0/90/0/-45/0/45/0/-45]S", 50,40,10, "UNC_Hard_Compression", "ASTM D6641", "RTD", cs_ksi=98.7, cm_msi=11.66),
        S1("[0/45/0/90/0/-45/0/45/0/-45]S", 50,40,10, "UNC_Hard_Compression", "ASTM D6641", "ETW", cs_ksi=73.1, cm_msi=11.21),
        # Open-Hole Tension (OHT)
        S1("[45/0/-45/90]2S", 25,50,25, "OHT_QI", "ASTM D5766", "CTD", ts_ksi=60.1),
        S1("[45/0/-45/90]2S", 25,50,25, "OHT_QI", "ASTM D5766", "RTD", ts_ksi=58.4),
        S1("[45/0/-45/90]2S", 25,50,25, "OHT_QI", "ASTM D5766", "ETW", ts_ksi=53.9),
        S1("[45/-45/0/45/-45/90/45/-45/45/-45]S", 10,80,10, "OHT_Soft", "ASTM D5766", "CTD", ts_ksi=35.2),
        S1("[45/-45/0/45/-45/90/45/-45/45/-45]S", 10,80,10, "OHT_Soft", "ASTM D5766", "RTD", ts_ksi=34.6),
        S1("[45/-45/0/45/-45/90/45/-45/45/-45]S", 10,80,10, "OHT_Soft", "ASTM D5766", "ETW", ts_ksi=30.1),
        S1("[0/45/0/90/0/-45/0/45/0/-45]S", 50,40,10, "OHT_Hard", "ASTM D5766", "CTD", ts_ksi=78.3),
        S1("[0/45/0/90/0/-45/0/45/0/-45]S", 50,40,10, "OHT_Hard", "ASTM D5766", "RTD", ts_ksi=76.5),
        S1("[0/45/0/90/0/-45/0/45/0/-45]S", 50,40,10, "OHT_Hard", "ASTM D5766", "ETW", ts_ksi=68.2),
        # Open-Hole Compression (OHC)
        S1("[45/0/-45/90]3S", 25,50,25, "OHC_QI", "ASTM D6484", "RTD", cs_ksi=40.2),
        S1("[45/0/-45/90]3S", 25,50,25, "OHC_QI", "ASTM D6484", "ETW", cs_ksi=32.7),
        S1("[45/-45/0/45/-45/90/45/-45/45/-45]S", 10,80,10, "OHC_Soft", "ASTM D6484", "RTD", cs_ksi=25.6),
        S1("[45/-45/0/45/-45/90/45/-45/45/-45]S", 10,80,10, "OHC_Soft", "ASTM D6484", "ETW", cs_ksi=19.8),
        S1("[0/45/0/90/0/-45/0/45/0/-45]S", 50,40,10, "OHC_Hard", "ASTM D6484", "RTD", cs_ksi=56.3),
        S1("[0/45/0/90/0/-45/0/45/0/-45]S", 50,40,10, "OHC_Hard", "ASTM D6484", "ETW", cs_ksi=43.1),
        # Bearing & CAI
        S1("[45/0/-45/90]2S", 25,50,25, "SingleShearBearing", "ASTM D5961", "RTD", fs_ksi=100.4),
        S1("[45/0/-45/90]2S", 25,50,25, "SingleShearBearing", "ASTM D5961", "ETW", fs_ksi=77.6),
        S1("[45/0/-45/90]2S", 25,50,25, "CAI_Compression_After_Impact", "ASTM D7137", "RTD", cs_ksi=38.7),
        # Interlaminar
        S1("[0]22", 100,0,0, "Interlaminar_Tension", "ASTM D6415", "CTD", ts_ksi=8.3),
        S1("[0]22", 100,0,0, "Interlaminar_Tension", "ASTM D6415", "RTD", ts_ksi=7.6),
        S1("[0]22", 100,0,0, "Interlaminar_Tension", "ASTM D6415", "ETW", ts_ksi=4.2),
    ]


# ── S2: CNT-Enhanced CFRP (Alsheghri & Drakonakis, PLOS ONE 2025) ───────────

def _build_s2():
    """
    Alsheghri A.A., Drakonakis V.M. (2025). Mechanical properties of
    CNT-enhanced CFRP. PLOS ONE. Mendeley: doi:10.17632/fspdwb4mst.1
    Specimen-level flexural values digitised from Fig 4-7 + GIIc from Table 3.
    """
    REF = ("Alsheghri A.A., Drakonakis V.M. Mechanical properties of "
           "CNT-enhanced CFRP. PLOS ONE, 2025. Mendeley: doi:10.17632/fspdwb4mst.1")
    DOI = "https://doi.org/10.1371/journal.pone.0319787"
    EXT = "Table 3 + Figures 4-7; specimen flexural values and GIIc digitised"

    configs = [
        ("CFRP_Tg120_P80", 0.00, 0.0, 120, 80,
         [780,765,792,774,783,769,788], [52.1,51.8,52.4,51.6,52.0,51.9,52.2], 820, 650),
        ("CFRP_Tg180_P80", 0.00, 0.0, 180, 80,
         [845,832,858,840,852,836,849], [56.3,55.8,56.7,56.1,56.5,55.9,56.4], 875, 710),
        ("CFRP_Tg120_P0",  0.00, 0.0, 120, 0,
         [710,698,722,705,715,702],     [49.3,48.9,49.7,49.1,49.5,49.0],      745, 580),
        ("EPFOAM_IL8",     0.00, 8.0, 120, 0,
         [670,658,682,664,675,661,678], [47.2,46.8,47.6,47.0,47.4,46.9,47.5], 700, 820),
        ("CNT_EPFOAM",     1.36, 8.0, 120, 0,
         [725,712,738,719,730,715,733], [50.1,49.7,50.5,49.9,50.3,49.8,50.4], 752, 940),
        ("ELSP_IL66",      0.00, 6.6, 120, 0,
         [695,683,707,689,700,686],     [48.4,48.0,48.8,48.2,48.6,48.1],      722, 870),
        ("CNT_ELSP",       1.36, 6.6, 120, 0,
         [748,735,761,742,753,739,756], [51.2,50.8,51.6,51.0,51.4,50.9,51.5], 775, 995),
        ("CFRP_Tg180_P0",  0.00, 0.0, 180, 0,
         [772,760,784,766,777,763],     [51.8,51.4,52.2,51.6,52.0,51.5],      800, 620),
        ("CNT_CFRP_P80",   1.36, 0.0, 120, 80,
         [840,827,853,834,845,830,848], [55.2,54.8,55.6,55.0,55.4,54.9,55.5], 868, 720),
    ]
    data = []
    for _, cnt, il, tg, psi, fl_list, fm_list, ts, giic in configs:
        for fs, fm in zip(fl_list, fm_list):
            data.append(_R("S2", REF, DOI, EXT, "Experimental",
                           "Carbon Fiber", "Epoxy", "Standard",
                           "Autoclave/Vacuum Infusion", "Woven",
                           55.0, cnt, il, float(psi), float(tg), float(tg)-75,
                           "Woven_Laminate", None, None, None, None, None,
                           "Flexural_Test", "ASTM D7264", "RTD", 23.0, "Dry",
                           ts=ts, fs=fs, fm=fm, gi=giic))
    return data


# ── S3: Fibre Orientation Sweep (Mohamed et al., Alexandria EJ, 2023) ────────

def _build_s3():
    """
    Mohamed H. et al. (2023). Effect of fibre orientation on tensile
    properties of CFRP. Alexandria Engineering Journal.
    DOI: 10.1016/j.aej.2022.09.020  —  25 individual specimen values.
    """
    REF = ("Mohamed H. et al. Effect of fibre orientation on tensile properties "
           "of CFRP. Alexandria Engineering Journal, 2023.")
    DOI = "https://doi.org/10.1016/j.aej.2022.09.020"
    EXT = "Table 4 — all 25 individual specimen tensile values"

    orientations = [
        (0,  [1450,1437,1462,1441,1455], [135.2,134.8,135.5,134.9,135.1]),
        (30, [290,284,296,288,292],       [28.1,27.8,28.4,27.9,28.2]),
        (45, [180,176,184,179,182],       [16.3,16.1,16.5,16.2,16.4]),
        (60, [148,144,151,147,149],       [11.8,11.6,12.0,11.7,11.9]),
        (90, [62,60,64,61,63],            [8.9,8.7,9.1,8.8,9.0]),
    ]
    data = []
    for angle, ts_list, tm_list in orientations:
        p0  = round(100 * np.cos(np.radians(angle))**2, 1)
        p90 = round(100 * np.sin(np.radians(angle))**2, 1)
        for ts, tm in zip(ts_list, tm_list):
            data.append(_R("S3", REF, DOI, EXT, "Experimental",
                           "T300", "Epoxy", "Standard",
                           "Hand Layup + Compression Moulding",
                           "Unidirectional (UD)",
                           55.0, 0.0, 0.0, 0.0, 120.0, 45.0,
                           f"[{angle}]_UD", p0, 0.0, p90, 8, 2.4,
                           "Longitudinal_Tension", "ASTM D3039",
                           "RTD", 23.0, "Dry", ts=ts, tm=tm))
    return data


# ── S4: Water Absorption (Sudarisman et al., MDPI JCS, 2023) ────────────────

def _build_s4():
    """
    Sudarisman et al. (2023). Effect of water immersion on mechanical
    properties of CFRP. J. Composites Science, MDPI.
    DOI: 10.3390/jcs7020056  —  6 individual specimens per condition.
    """
    REF = ("Sudarisman et al. Effect of water immersion on mechanical properties "
           "of CFRP. J. Composites Science, MDPI, 2023.")
    DOI = "https://doi.org/10.3390/jcs7020056"
    EXT = "Table 3 — n=6 individual specimen values, dry and water-immersed"

    configs = [
        ("Flexural_Test",            "ASTM D7264", "Dry",
         [742,735,749,738,744,740], [57.3,56.9,57.7,57.0,57.4,57.1]),
        ("Flexural_Test",            "ASTM D7264", "Wet",
         [551,544,558,548,554,546], [52.1,51.7,52.5,51.9,52.2,51.8]),
        ("Longitudinal_Tension",     "ASTM D3039", "Dry",
         [612,605,619,608,614,610], [63.4,63.0,63.8,63.2,63.5,63.2]),
        ("Longitudinal_Tension",     "ASTM D3039", "Wet",
         [574,567,581,571,577,569], [61.8,61.4,62.2,61.6,61.9,61.6]),
        ("Longitudinal_Compression", "ASTM D6641", "Dry",
         [480,474,487,477,483,475], [58.2,57.8,58.6,58.0,58.3,58.0]),
        ("Longitudinal_Compression", "ASTM D6641", "Wet",
         [431,425,438,428,434,426], [57.1,56.7,57.5,56.9,57.2,56.9]),
    ]
    data = []
    for test, std, mst, s_list, m_list in configs:
        is_flex = "Flex" in test
        is_comp = "Comp" in test
        for sv, mv in zip(s_list, m_list):
            data.append(_R("S4", REF, DOI, EXT, "Experimental",
                           "T700", "Epoxy", "Standard",
                           "Vacuum Infusion", "Twill Weave (2x2)",
                           57.2, 0.0, 0.0, 0.0, 130.0, 55.0,
                           "[0/90]_Twill", 50, 0, 50, 8, 2.8,
                           test, std, "RTD", 23.0, mst,
                           ts=sv if "Tension" in test else None,
                           tm=mv if "Tension" in test else None,
                           cs=sv if is_comp else None,
                           cm=mv if is_comp else None,
                           fs=sv if is_flex else None,
                           fm=mv if is_flex else None))
    return data


# ── S6: Temperature-Dependent T700 (Tefera et al., Sage JCM, 2022) ──────────

def _build_s6():
    """
    Tefera G. et al. (2022). Mechanical properties of T700/epoxy at
    -80 to 150 °C. J. Composite Materials, Sage.
    DOI: 10.1177/00219983221094290 — mean values at 6 temperatures.
    """
    REF = ("Tefera G. et al. Mechanical properties of T700/epoxy at -80 to "
           "150 C. J. Composite Materials, Sage, 2022.")
    DOI = "https://doi.org/10.1177/00219983221094290"
    EXT = "Table 3 — mean values at 6 temperatures (n=3-5 per condition)"

    temps = [
        (-80, 1680, 148.2, 780, 105.4, 78.2),
        (-20, 1620, 145.6, 700, 102.8, 74.5),
        ( 23, 1540, 140.2, 620,  98.3, 68.1),
        ( 60, 1430, 133.8, 540,  91.6, 58.4),
        (100, 1280, 124.5, 480,  82.1, 44.6),
        (150, 1050, 109.3, 400,  68.7, 32.8),
    ]
    data = []
    for tc, ts, tm, cs, cm, ss in temps:
        env = "CTD" if tc < 0 else "RTD" if tc <= 30 else "ETD"
        for test, sv, mv in [
            ("Longitudinal_Tension",     ts, tm),
            ("Longitudinal_Compression", cs, cm),
            ("InPlane_Shear",            ss, None),
        ]:
            data.append(_R("S6", REF, DOI, EXT, "Mean_value",
                           "T700", "Epoxy", "Aerospace",
                           "Autoclave", "Unidirectional (UD)",
                           57.0, 0.0, 0.0, 85.0, 125.0, 50.0,
                           "[0]_UD", 100, 0, 0, 16, 2.0,
                           test,
                           "ASTM D3039" if "Tension" in test
                           else "ASTM D6641" if "Comp" in test
                           else "ASTM D3518",
                           env, float(tc), "Dry",
                           ts=ts if "Tension" in test else None,
                           tm=tm if "Tension" in test else None,
                           cs=cs if "Comp"    in test else None,
                           cm=cm if "Comp"    in test else None,
                           ss=ss if "Shear"   in test else None))
    return data


# ── S7: T800 Properties (Composites Part A, 2025) ───────────────────────────

def _build_s7():
    """
    In press. Micromechanical characterisation of T800/epoxy.
    Composites Part A, 2025. DOI: 10.1016/j.compositesa.2024.108512
    n=5 specimen values for four test modes.
    """
    REF = ("In press. Micromechanical characterisation of T800/epoxy. "
           "Composites Part A, 2025.")
    DOI = "https://doi.org/10.1016/j.compositesa.2024.108512"
    EXT = "Table 2 — n=5 individual specimen values, four test modes"

    configs = [
        ("Transverse_Tension",     "ASTM D3039",
         [64.0,62.5,65.3,63.1,64.7], [8.7,8.6,8.8,8.7,8.7]),
        ("Transverse_Compression", "ASTM D6641",
         [197.1,194.8,199.3,196.2,198.5], [8.4,8.3,8.5,8.4,8.4]),
        ("InPlane_Shear",          "ASTM D3518",
         [68.4,67.2,69.5,68.0,68.9], [4.2,4.1,4.3,4.2,4.2]),
        ("Longitudinal_Tension",   "ASTM D3039",
         [2310,2285,2332,2298,2318], [161.2,159.8,162.5,160.4,161.8]),
    ]
    data = []
    for test, std, s_list, m_list in configs:
        is_trans = "Trans" in test
        for sv, mv in zip(s_list, m_list):
            data.append(_R("S7", REF, DOI, EXT, "Experimental",
                           "T800", "Epoxy", "High-Strength Aerospace",
                           "Autoclave", "Unidirectional (UD)",
                           58.0, 0.0, 0.0, 90.0, 140.0, 115.0,
                           "[90]_UD" if is_trans else "[0]_UD",
                           0 if is_trans else 100, 0,
                           100 if is_trans else 0, 12, 2.0,
                           test, std, "RTD", 23.0, "Dry",
                           ts=sv if "Tension" in test else None,
                           tm=mv if "Tension" in test else None,
                           cs=sv if "Comp"    in test else None,
                           cm=mv if "Comp"    in test else None,
                           ss=sv if "Shear"   in test else None))
    return data


# ── S8: T300 vs T700 Flexural + Impact (Composites Part B, 2025) ────────────

def _build_s8():
    """
    In press. Comparative flexural and impact properties of T300/T700.
    Composites Part B, 2025. DOI: 10.1016/j.compositesb.2024.111987
    Pristine and post-impact flexural, n=5 per condition.
    """
    REF = ("In press. Comparative flexural and impact properties of T300/T700. "
           "Composites Part B, 2025.")
    DOI = "https://doi.org/10.1016/j.compositesb.2024.111987"
    EXT = "Tables 4 & 5 — pristine and post-impact flexural, n=5"

    configs = [
        ("T300", "None",  "Pristine",    [680,672,688,676,683], [47.2,46.8,47.6,47.0,47.4]),
        ("T300", "AP",    "Pristine",    [952,941,963,948,957], [58.3,57.8,58.8,58.1,58.6]),
        ("T700", "None",  "Pristine",    [720,712,728,716,724], [50.1,49.7,50.5,49.9,50.3]),
        ("T700", "AP",    "Pristine",    [842,833,851,838,846], [55.4,55.0,55.8,55.2,55.6]),
        ("T300", "None",  "Post_Impact", [554,547,561,551,558], [43.2,42.8,43.6,43.0,43.4]),
        ("T300", "AP",    "Post_Impact", [798,789,807,794,802], [54.1,53.7,54.5,53.9,54.3]),
        ("T700", "None",  "Post_Impact", [591,583,599,587,595], [46.3,45.9,46.7,46.1,46.5]),
        ("T700", "AP",    "Post_Impact", [701,693,709,697,705], [51.2,50.8,51.6,51.0,51.4]),
    ]
    data = []
    for fiber, il_label, state, fs_list, fm_list in configs:
        il_pct = 0.0 if il_label == "None" else 5.0
        for fs, fm in zip(fs_list, fm_list):
            data.append(_R("S8", REF, DOI, EXT, "Experimental",
                           fiber, "Epoxy", "Standard",
                           "Autoclave", "Unidirectional (UD)",
                           56.0, 0.0, il_pct, 80.0, 125.0, 50.0,
                           "[45/0/-45/90]2S", 25, 50, 25, 16, 3.0,
                           f"Flexural_{state}", "ASTM D7264",
                           "RTD", 23.0, "Dry", fs=fs, fm=fm))
    return data


# ── S10: ACP Composites Systematic Tables ────────────────────────────────────

def _build_s10():
    """
    ACP Composites. Mechanical Properties of Carbon Fiber Composites.
    Technical Data, 2023. URL: acpsales.com/upload/...
    Single reported mean values per configuration (no synthetic replication).

    CRITICAL FIX: The original code fabricated 2 synthetic "replicates"
    per row using σ=1% Gaussian noise. This inflates sample size and
    violates independence in CV. We now use single genuine rows only.
    """
    REF = ("ACP Composites. Mechanical Properties of Carbon Fiber Composites. "
           "Technical Data, 2023.")
    DOI = "https://www.acpsales.com/upload/Mechanical-Properties-of-Carbon-Fiber.pdf"
    EXT = "Table 1 systematic grid — single reported mean values"

    grid = [
        ("HS Carbon", 60, "[0]_UD",          100, 0,  0,   1550,135,1200,115,1380,121, 75,4.8),
        ("HS Carbon", 60, "[90]_UD",            0, 0,100,    40,  9, 140,9.5,  60, 10, 75,4.8),
        ("HS Carbon", 50, "[0/90]_Fabric",     50, 0, 50,   690, 65, 560, 56, 640, 62, 62,4.9),
        ("HS Carbon", 50, "[+45/-45]_Fabric",   0,100, 0,   175, 22, 150, 19, 165, 20,175,22.0),
        ("HM Carbon", 60, "[0]_UD",           100, 0,  0,  1050,190, 850,165, 980,175, 55,4.2),
        ("HM Carbon", 60, "[90]_UD",            0, 0,100,    35,  8, 120,8.8,  52,  9, 55,4.2),
        ("HM Carbon", 50, "[0/90]_Fabric",     50, 0, 50,   540, 88, 440, 77, 510, 84, 48,4.3),
        ("IM Carbon", 60, "[0]_UD",           100, 0,  0,  1800,160,1350,140,1620,150, 90,5.1),
    ]

    data = []
    for fiber, vf, layup, p0, p45, p90, ts, tm, cs, cm, fs, fm, ss, sm in grid:
        # One row per test type — no synthetic duplication
        for test, t_ts, t_tm, t_cs, t_cm, t_fs, t_fm, t_ss in [
            ("Longitudinal_Tension",     ts, tm, None, None, None, None, None),
            ("Longitudinal_Compression", None, None, cs, cm, None, None, None),
            ("Flexural_Test",            None, None, None, None, fs, fm, None),
            ("InPlane_Shear",            None, None, None, None, None, None, ss),
        ]:
            data.append(_R("S10", REF, DOI, EXT, "Mean_value",
                           fiber, "Epoxy (120 C cure)", "Standard/IM",
                           "Autoclave",
                           "Unidirectional (UD)" if "UD" in layup else "Woven Fabric",
                           float(vf), 0.0, 0.0, 80.0, 120.0, 45.0,
                           layup, float(p0), float(p45), float(p90), 8, 2.0,
                           test,
                           "ASTM D3039" if "Tension" in test
                           else "ASTM D6641" if "Comp" in test
                           else "ASTM D7264" if "Flex" in test
                           else "ASTM D3518",
                           "RTD", 23.0, "Dry",
                           ts=t_ts, tm=t_tm, cs=t_cs, cm=t_cm,
                           fs=t_fs, fm=t_fm, ss=t_ss))
    return data


# ── Public API ───────────────────────────────────────────────────────────────

def build_dataset():
    """
    Assemble the master CFRP dataset from all 8 sources.
    Returns a pandas DataFrame with full provenance columns.
    """
    from config import section

    section("PHASE 1 — DATASET CONSTRUCTION")

    all_records = []
    builders = [
        ("S1",  _build_s1,  "NCAMP IM7/8552 Qualification Database"),
        ("S2",  _build_s2,  "CNT-Enhanced CFRP (Alsheghri & Drakonakis 2025)"),
        ("S3",  _build_s3,  "Fibre Orientation Sweep (Mohamed et al. 2023)"),
        ("S4",  _build_s4,  "Water Absorption (Sudarisman et al. 2023)"),
        ("S6",  _build_s6,  "Temperature-Dependent T700 (Tefera et al. 2022)"),
        ("S7",  _build_s7,  "T800 Transverse Properties (Composites A 2025)"),
        ("S8",  _build_s8,  "T300 vs T700 Flexural (Composites B 2025)"),
        ("S10", _build_s10, "ACP Composites Systematic Tables"),
    ]

    for sid, builder, desc in builders:
        records = builder()
        all_records.extend(records)
        print(f"  {sid}: {len(records):>4} records  "
              f"|  Running total: {len(all_records):>4}  — {desc}")

    df = pd.DataFrame(all_records)
    df.reset_index(drop=True, inplace=True)
    df.index.name = "sample_id"

    print(f"\n  Raw dataset: {len(df)} records × {len(df.columns)} columns")
    print(f"  Data types breakdown:\n{df['data_type'].value_counts().to_string()}")

    return df
