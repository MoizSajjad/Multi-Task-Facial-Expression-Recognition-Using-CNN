import numpy as np

def _mask_valid(a):
    # valid if != -2 (assignment convention)
    return np.isfinite(a) & (a != -2)

def rmse(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum()==0: return float("nan")
    return float(np.sqrt(np.mean((y_true[m]-y_pred[m])**2)))

def pearsonr(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2: return float("nan")
    yt, yp = y_true[m], y_pred[m]
    yt = yt - yt.mean(); yp = yp - yp.mean()
    denom = (np.sqrt((yt**2).sum()) * np.sqrt((yp**2).sum()))
    return float((yt @ yp) / denom) if denom else float("nan")

def ccc(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2: return float("nan")
    yt, yp = y_true[m], y_pred[m]
    mu_t, mu_p = yt.mean(), yp.mean()
    vt, vp = yt.var(), yp.var()
    cov = np.mean((yt-mu_t)*(yp-mu_p))
    denom = vt + vp + (mu_t - mu_p)**2
    return float((2*cov)/denom) if denom else float("nan")

def sagr(y_true, y_pred):
    # Sign Agreement: proportion of samples where sign matches (treat 0 as 0)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum()==0: return float("nan")
    s_true = np.sign(y_true[m])
    s_pred = np.sign(y_pred[m])
    return float((s_true == s_pred).mean())

def va_metrics(y_true_va, y_pred_va):
    y_true_va = np.asarray(y_true_va)  # [N,2]
    y_pred_va = np.asarray(y_pred_va)
    vt, at = y_true_va[:,0], y_true_va[:,1]
    vp, ap = y_pred_va[:,0], y_pred_va[:,1]
    return {
        "val_rmse": rmse(vt, vp),
        "aro_rmse": rmse(at, ap),
        "val_pearson": pearsonr(vt, vp),
        "aro_pearson": pearsonr(at, ap),
        "val_ccc": ccc(vt, vp),
        "aro_ccc": ccc(at, ap),
        "val_sagr": sagr(vt, vp),
        "aro_sagr": sagr(at, ap),
    }
