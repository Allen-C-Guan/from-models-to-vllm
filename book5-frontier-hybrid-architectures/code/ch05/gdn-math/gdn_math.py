"""GDN数学教学件：纯CPU可跑，不含融合核、模型权重或性能声明。
输入布局 (..., N, d)，状态 (..., d_k, d_v)，前缀通常为(B,H)。
运行：source ./env.sh 后 python code/Book1-5-系统重写/sample-gdn/verify.py
"""
import torch


def recurrent(q, k, v, alpha, beta, state):
    """先忘、读预测、写残差、读输出；保持调用者的dtype。"""
    outputs, trace = [], []
    for t in range(q.shape[-2]):
        old = state
        faded = alpha[..., t, None, None] * old     # (..., d_k, d_v)
        prediction = (k[..., t, :, None] * faded).sum(-2)  # (..., d_v)
        residual = beta[..., t, None] * (v[..., t, :] - prediction)
        state = faded + k[..., t, :, None] * residual[..., None, :]
        output = (q[..., t, :, None] * state).sum(-2)
        outputs.append(output)
        trace.append(dict(old=old, faded=faded, prediction=prediction,
                          residual=residual, state=state, output=output))
    return torch.stack(outputs, -2), state, trace


def coefficients(alpha):
    """用有限乘积构造D，避免alpha=0时log或比值的奇点；教学用C²循环。"""
    c = alpha.shape[-1]
    one, zero = torch.ones_like(alpha[..., 0]), torch.zeros_like(alpha[..., 0])
    rows = []
    for i in range(c):
        # 从对角向左递推同一行，每格只做一次乘法；总计O(C²)，不重复区间prod。
        row = [zero] * c
        survival = one
        for j in range(i, -1, -1):
            row[j] = survival
            survival = survival * alpha[..., j]
        rows.append(torch.stack(row, -1))
    decay = torch.stack(rows, -2)  # (..., C, C)，含对角的下三角
    gamma = alpha.cumprod(-1)    # (..., C)，块入口存活率
    eta = decay[..., -1, :]      # 块尾存活率就是D的最后一行
    return decay, gamma, eta


def block(q, k, v, alpha, beta, state):
    """一个块的两个右端求解；返回调试量。正文中的A、U、W、R均在此。"""
    decay, gamma, eta = coefficients(alpha)
    gram = k @ k.transpose(-1, -2)             # (..., C, C)
    lower = torch.tril(beta[..., :, None] * decay * gram, diagonal=-1)
    system = torch.eye(k.shape[-2], dtype=k.dtype, device=k.device) + lower
    rhs_v = beta[..., :, None] * v            # (..., C, d_v)
    rhs_k = (beta * gamma)[..., :, None] * k  # (..., C, d_k)
    u = torch.linalg.solve_triangular(system, rhs_v, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(system, rhs_k, upper=False, unitriangular=True)
    residuals = u - w @ state                # (..., C, d_v)，实际净写入R
    weights = decay * (q @ k.transpose(-1, -2))
    outputs = (gamma[..., :, None] * q) @ state + weights @ residuals
    final = gamma[..., -1, None, None] * state + k.transpose(-1, -2) @ (eta[..., :, None] * residuals)
    trace = dict(D=decay, gamma=gamma, eta=eta, L=lower, A=system,
                 U=u, W=w, R=residuals, P=weights, O=outputs, S_out=final)
    return outputs, final, trace


def chunkwise(q, k, v, alpha, beta, state, chunk=3):
    """顺序交接块状态；尾块直接缩短，不引入padding。"""
    outputs, trace = [], []
    for start in range(0, q.shape[-2], chunk):
        stop = start + chunk
        out, state, info = block(q[..., start:stop, :], k[..., start:stop, :],
                                 v[..., start:stop, :], alpha[..., start:stop],
                                 beta[..., start:stop], state)
        outputs.append(out)
        trace.append(info)
    return torch.cat(outputs, -2), state, trace


def affine(q, k, v, alpha, beta, state):
    """从原式直接矩阵乘，独立于residual计算路径。"""
    eye = torch.eye(k.shape[-1], dtype=k.dtype, device=k.device)
    outputs = []
    for t in range(q.shape[-2]):
        kt, vt = k[..., t, :], v[..., t, :]
        erase = eye - beta[..., t, None, None] * kt[..., :, None] * kt[..., None, :]
        state = alpha[..., t, None, None] * (erase @ state) + beta[..., t, None, None] * kt[..., :, None] * vt[..., None, :]
        outputs.append((q[..., t, :, None] * state).sum(-2))
    return torch.stack(outputs, -2), state


def toy():
    """正文统一构造例：有理数，非零入口；未加q的实现缩放。"""
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    k = t([[1, 0], [.6, .8], [.8, -.6]])
    return k.clone(), k, t([[5, 0], [0, 2], [3, 1]]), t([1, .5, .5]), t([1, 1, 1]), t([[2, 0], [0, 0]])
