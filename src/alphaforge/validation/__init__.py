"""Validation: IC metrics and overlap-honest inference (alphaDesign.md §7).

Curated public surface — downstream modules import from
:mod:`alphaforge.validation`, not its submodules.
"""

from alphaforge.validation.cpcv import CombinatorialPurgedCV
from alphaforge.validation.diversification import (
    DiversificationReport,
    diversification_report,
)
from alphaforge.validation.dsr import (
    EULER_MASCHERONI,
    DSRReport,
    deflated_sharpe_ratio,
    dsr_from_returns,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from alphaforge.validation.experiments import (
    DEFAULT_LOG_PATH,
    DEFAULT_SR_TRIALS_VARIANCE,
    ExperimentLog,
    ExperimentRecord,
    ExperimentUnion,
    config_hash,
)
from alphaforge.validation.metrics import (
    ICSummary,
    ic_summary,
    newey_west_tstat,
    non_overlapping,
    rank_ic,
)
from alphaforge.validation.pbo import PBOResult, pbo_cscv
from alphaforge.validation.sleeve_admission import (
    AdmissionReport,
    evaluate_sleeve_evidence,
    load_admission_contract,
)
from alphaforge.validation.splits import PurgedWalkForward

__all__ = [
    "DEFAULT_LOG_PATH",
    "DEFAULT_SR_TRIALS_VARIANCE",
    "EULER_MASCHERONI",
    "AdmissionReport",
    "CombinatorialPurgedCV",
    "DSRReport",
    "DiversificationReport",
    "ExperimentLog",
    "ExperimentRecord",
    "ExperimentUnion",
    "ICSummary",
    "PBOResult",
    "PurgedWalkForward",
    "config_hash",
    "deflated_sharpe_ratio",
    "diversification_report",
    "dsr_from_returns",
    "evaluate_sleeve_evidence",
    "expected_max_sharpe",
    "ic_summary",
    "load_admission_contract",
    "newey_west_tstat",
    "non_overlapping",
    "pbo_cscv",
    "probabilistic_sharpe_ratio",
    "rank_ic",
]
