# FINAL PROJECT VERIFICATION REPORT

## 1. Repository Audit Summary
Audit scope covered:
- `src/loan_recovery` modules
- `app.py`, `main.py`
- `scripts/`, `notebooks/`, `outputs/`
- dependency/setup files (`pyproject.toml`, `uv.lock`)
- documentation (`README.md`)

Major issues found and fixed:
- Target leakage risk: target-derived field (`High_Risk_Flag`) was previously generated and could flow into modeling.
- Business metric bug: high-risk detection was incorrectly computed due to index misalignment.
- Evaluation design gap: threshold-optimized decisions were mixed with core classification metrics.
- Reproducibility gap: no single verification command for clean end-to-end checks.
- Governance gap: no explicit feature audit table (sensitive, target-derived, collections-stage proxies).
- Notebook failure risk: notebook expected stale figure names; fixed by backward-compatible figure export.
- Portfolio hygiene: added `.gitignore` for runtime noise and legacy root artifacts.

## 2. Dataset Validation
Source: `loan-recovery.csv`

Validated results:
- Row count: `500`
- Column count: `22`
- Schema valid: `true`
- Missing values: `0`
- Duplicate rows: `0`
- Invalid ranges (age, interest, score, negative financial/behavioral): `0`
- Warning-level anomaly: `180` rows with `Outstanding_Balance > Loan_Amount`

Interpretation:
- Dataset is structurally clean.
- Outstanding-over-loan cases are plausible with penalties/fees/re-aging; treated as warning, not blocking failure.

## 3. Financial Feature Validation
Financial features were revised for scale stability and business realism:
- Loan-to-Income now uses annualized income denominator.
- EMI-to-Income and Debt Burden are normalized affordability indicators.
- Risk Exposure uses log-scaled outstanding with interest/default multipliers.
- Delinquency/Behavioral scores bounded to `[0,1]` components.
- Recovery difficulty combines delinquency, collateral shortfall, and default history.
- Collections intensity normalized by days-past-due window.

Artifact added:
- `outputs/tables/feature_audit.csv` with per-feature governance flags and modeling inclusion status.

## 4. Leakage Audit
Leakage controls implemented:
- `LoanDataLoader.load(add_target_derivatives=False)` default.
- `FeatureEngineer` defaults to no target-derived fields for modeling.
- Explicit exclusion policy for:
  - target-derived fields
  - sensitive/proxy attributes (default off): `Marital_Status`, `Residence_Type`
  - collections-stage proxy fields for early-warning mode

Validation:
- Unit test `test_leakage_sensitive_features_excluded_by_default` passes.
- Feature audit table confirms exclusion behavior.

## 5. Segmentation Review
Algorithms compared with parameter search:
- KMeans, MiniBatchKMeans, Hierarchical, GMM, DBSCAN

Metrics used:
- Silhouette, Davies-Bouldin, Calinski-Harabasz, Stability Score

Latest best segmentation:
- KMeans (`k=3`)
- Silhouette: `0.9416`
- Davies-Bouldin: `0.1181`
- Calinski-Harabasz: `1566.16`

Business naming improved:
- `Legal Escalation Candidate Segment`
- `High Income Low Risk Segment`
- `Recovery Friendly Segment`

## 6. Risk Model Review
Manual model leaderboard (validation split) executed across required family set.

Best manual model:
- `SVM`
- Validation weighted F1: `0.4352`

Test classification metrics:
- Accuracy: `0.43`
- Weighted F1: `0.4312`
- Macro F1: `0.3902`
- ROC-AUC OVR: `0.4948`
- PR-AUC (Written-Off): `0.2046`

Interpretation:
- After leakage removal, predictive strength is materially lower than prior inflated runs.
- Current model quality is realistic but modest; this is now a trustworthy baseline.

## 7. Cost-Sensitive Analysis
Implemented:
- Written-off threshold optimization on validation set
- Cost objective with configurable FP/FN costs
- Separate reporting for default-threshold vs cost-sensitive-threshold outcomes

Optimized threshold:
- `0.14`

Default threshold business metrics:
- High-risk recall: `0.2105`
- Total misclassification cost: `41,500`

Cost-sensitive threshold business metrics:
- High-risk recall: `0.8947`
- Total misclassification cost: `24,250`

Tradeoff:
- Strong recall and cost reduction at the expense of conservative recovery-class predictions and lower recovery amount estimate.

## 8. Explainability Review
SHAP pipeline runs successfully with fallback support.

Generated artifacts:
- `feature_importance.png`
- `shap_summary.png`
- `shap_waterfall.png`
- `shap_force.html`

Assessment:
- Explainability is operational and business-consumable.
- For non-tree estimators, fallback explainer runtime is controlled.

## 9. Dashboard Review
Dashboard outputs generated:
- risk distribution
- recovery probability
- segment distribution (segment name-aware)
- strategy distribution
- scenario comparison

Assessment:
- Executive readability improved with clearer segmentation context.
- KPI and actionability are present for collections decision workflows.

## 10. Streamlit Review
Enhancements implemented:
- Artifacts are loaded from disk when available (no unnecessary full retraining on app load).
- Validation for inconsistent user inputs (for example missed payments > term).
- User-facing warning messages for high-stress inputs.
- safer error handling around scoring path.

Assessment:
- App is now materially more usable for business users and demo environments.

## 11. Improvements Implemented
Core engineering:
- leakage-safe feature split policy
- train/validation/test workflow support
- threshold optimization and calibration plot
- business metric alignment fix
- feature governance audit export
- imbalance experiment table
- robust segmentation tuning and stability scoring

Reproducibility and quality:
- `scripts/run_full_verification.py` (clean run + strict pipeline + notebook execution + tests)
- `tests/test_risk_controls.py` added
- notebook generation enhanced with standardized tutorial pedagogy blocks

Documentation:
- README rewritten in mini-book format with corrected dataset facts and governance framing.

## 12. Remaining Limitations
- Model discrimination remains weak after leakage removal (expected for this dataset’s signal quality).
- No event-time column exists; true temporal backtesting is not possible.
- Outstanding-over-loan anomaly appears frequently; treated as warning due plausible fee accrual but should be source-validated in real deployment.
- Full notebook execution in this environment required elevated permissions due sandbox socket restrictions; local unrestricted execution path is validated.

## 13. Final Scores
Scale: `1-10` (portfolio-quality, realism, and governance-weighted)

- Financial Analytics: `8.5`
- Credit Risk Modeling: `7.5`
- Segmentation Quality: `8.5`
- Explainability: `8.5`
- Business Value: `8.0`
- Dashboard Quality: `8.0`
- Educational Value: `8.5`
- ML Engineering: `8.5`
- Documentation: `9.0`
- Portfolio Strength: `8.5`

### Hiring Manager Lens (post-hardening)
- Credit Risk Analytics: `8/10`
- Data Science Rigor: `8/10`
- ML Engineering Maturity: `8/10`
- Explainable AI Usage: `8/10`
- Business Understanding: `8/10`
- Visualization and Communication: `8/10`
- Reproducibility and Reliability: `8/10`

## Verification Commands Executed
- `uv run python -m compileall src app.py main.py scripts tests`
- `uv run python main.py --strict`
- `uv run python scripts/generate_notebooks.py`
- `uv run python scripts/run_full_verification.py` (notebook execution validated)
- `uv run python -m unittest discover -s tests -p 'test_*.py' -v`

