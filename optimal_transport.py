"""Historical log-domain Sinkhorn solver."""
import torch

@torch.no_grad()
def sinkhorn_log_stabilized(C, mu, nu, epsilon=0.1, max_iter=100, tol=1e-06, normalize_iters=20, marginal_tol=1e-06, debug=True):
    """Return a transport plan for cost C and unit-sum marginals using log-domain Sinkhorn."""
    device = C.device
    dtype = C.dtype
    eps = 1e-12
    if torch.isnan(C).any() or torch.isinf(C).any():
        raise ValueError('Sinkhorn input C contains NaN/Inf')
    if mu.dim() != 1 or nu.dim() != 1:
        raise ValueError('mu and nu must be 1-D tensors')
    (B, K) = C.shape
    if mu.numel() != B or nu.numel() != K:
        raise ValueError(f'Shape mismatch: C {C.shape}, mu {mu.shape}, nu {nu.shape}')
    u = torch.zeros(B, device=device, dtype=dtype)
    v = torch.zeros(K, device=device, dtype=dtype)
    log_mu = torch.log(mu + eps)
    log_nu = torch.log(nu + eps)
    for it in range(max_iter):
        lse_cols = torch.logsumexp((-C + v.unsqueeze(0)) / epsilon, dim=1)
        u_new = epsilon * (log_mu - lse_cols)
        lse_rows = torch.logsumexp((-C + u_new.unsqueeze(1)) / epsilon, dim=0)
        v_new = epsilon * (log_nu - lse_rows)
        if torch.max(torch.abs(u_new - u)) < tol and torch.max(torch.abs(v_new - v)) < tol:
            (u, v) = (u_new, v_new)
            break
        (u, v) = (u_new, v_new)
    log_T = (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon
    if torch.isnan(log_T).any() or torch.isinf(log_T).any():
        if debug:
            try:
                print('[Sinkhorn debug] log_T contained NaN/Inf before clamp')
                print('u stats:', u.min().item(), u.max().item(), u.mean().item())
                print('v stats:', v.min().item(), v.max().item(), v.mean().item())
            except Exception:
                pass
        log_T = torch.clamp(log_T, -50.0, 50.0)
    T = torch.exp(log_T)
    T = torch.clamp(T, min=eps)
    for _ in range(normalize_iters):
        row_sum = T.sum(dim=1, keepdim=True)
        T = T * (mu.unsqueeze(1) / (row_sum + eps))
        col_sum = T.sum(dim=0, keepdim=True)
        T = T * (nu.unsqueeze(0) / (col_sum + eps))
        row_err = torch.max(torch.abs(T.sum(dim=1) - mu))
        col_err = torch.max(torch.abs(T.sum(dim=0) - nu))
        if row_err < marginal_tol and col_err < marginal_tol:
            break
    if torch.isnan(T).any() or torch.isinf(T).any():
        if debug:
            print('[Sinkhorn debug] final T still contains NaN/Inf after clamp')
        raise ValueError('Sinkhorn produced NaN/Inf in final T')
    return T
