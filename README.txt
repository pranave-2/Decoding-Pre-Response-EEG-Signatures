EEG decision-prediction project - code bundle

notebooks/  Colab notebooks, run in order on the raw ds002739 data (Google Drive):
  Stage1_Window_Feasibility, Stage1B_Pooled_Class_Balance, Stage2_Epoch_Extraction,
  Stage4_PostResponse_Extraction, Stage4b_PostResponse_NoCrop (response-locked, channel alignment),
  Stage5_Personalized_Window_Extraction (per-subject windows)
scripts/    verification and analysis scripts (need the .npz outputs of the notebooks)
  minimal_lda_cv.py, bin_sweep.py, binning_alternatives.py, cross_subject_pooling.py,
  cpp_rt_link.py, verify_retention_independent.py, stage4_diagnostic.txt,
  personalized_vs_fixed.py (personalized vs fixed window comparison; run was only partly finished)
demo/       live EEG demo (pygame + pylsl) with simulator; see demo/README.md
paper/      LaTeX source, figures and PDF of the manuscript
Not included: spatial_integration*.py, personalized_window_plan.py, sanity_check_bands.py,
cpp_detail.py (their files were lost in a sandbox reset).
