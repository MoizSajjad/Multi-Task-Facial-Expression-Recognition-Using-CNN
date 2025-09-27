import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

def multiclass_metrics(y_true, y_pred, y_proba=None, num_classes=8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "conf_mat": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        Y = label_binarize(y_true, classes=list(range(num_classes)))
        try:
            out["roc_auc_macro"] = float(roc_auc_score(Y, y_proba, average="macro", multi_class="ovr"))
        except Exception:
            out["roc_auc_macro"] = float("nan")
        try:
            out["pr_auc_macro"] = float(average_precision_score(Y, y_proba, average="macro"))
        except Exception:
            out["pr_auc_macro"] = float("nan")
    return out

def krippendorff_alpha_nominal(y1, y2, num_classes=8):
    """
    Krippendorff's alpha (nominal) for two 'coders': y1 (ground-truth) and y2 (predictions).
    NOTE: Proper alpha is for >=2 coders; for 2 complete coders this reduces to a function
    of their coincidence matrix. We provide a standard nominal-alpha computation.
    """
    y1 = np.asarray(y1); y2 = np.asarray(y2)
    assert y1.shape == y2.shape
    C = np.zeros((num_classes, num_classes), dtype=float)
    for a, b in zip(y1, y2):
        if a is None or b is None: continue
        a = int(a); b = int(b)
        C[a, b] += 1
        C[b, a] += 1  # coincidence matrix counts both orders

    n = C.sum()
    if n <= 1: return float("nan")
    # Observed disagreement Do (nominal → δ=1 when i!=j)
    Do = (C.sum() - np.trace(C)) / (n)
    # Expected disagreement De based on marginals
    m = C.sum(axis=1)
    De = (n * n - (m @ m)) / (n * n)
    if De == 0: return 1.0
    return 1.0 - Do / De
