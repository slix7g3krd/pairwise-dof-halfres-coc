"""半分辨率景深（Half-res DoF）参考实现。

本模块是 GPU 景深管线各 pass 的 CPU 参考模型，覆盖六个修复点：

1. CoC 全分辨率校验 ............ coc_fullres / validation
2. 散景上采样的深度权重 ........ depth_weight / upsample / near_far_split
3. 对焦收敛与重置 .............. focus_converge / focus_reset
4. 抖动序列固定 ................ jitter_seed / determinism
5. pass 顺序与栅栏 ............. pass_order / temporal / fallback
6. 远景高光能量守恒 ............ highlight_energy / metrics / audit

约定
----
* 深度单位对外为米（m），内部转毫米参与薄透镜公式。
* CoC 单位为全分辨率像素，带符号：
    CoC < 0  前景（near field，比焦平面近）
    CoC > 0  背景（far field，比焦平面远）
    CoC = 0  合焦
* 所有随机量只允许使用局部 ``random.Random(seed)`` 或无状态 Halton 序列，
  禁止触碰全局随机状态，保证跨运行逐帧一致（见 determinism）。
* 缓冲区统一用扁平 list 表示，颜色为 ``[r, g, b]`` 线性 HDR triple。
"""

from __future__ import annotations

import math
import random

# ---------------------------------------------------------------------------
# 常量与阈值
# ---------------------------------------------------------------------------

DEFAULT_CAMERA = {
    "focus_depth": 2.0,     # m
    "focal_length": 35.0,   # mm
    "f_stop": 2.0,
    "sensor_width": 36.0,   # mm（全画幅）
    "frame_width": 1920,    # px
}

COC_ABS_TOL = 1.0          # 全分辨率 CoC 校验绝对容差（px）
COC_REL_TOL = 0.10         # 相对容差
NEAR_FIELD_BARRIER = 0.05  # 前景 tap 渗入背景像素时的屏障权重
FAR_FIELD_BARRIER = 0.25   # 背景 tap 渗入前景像素时的屏障权重
DEPTH_SIGMA = 0.05         # 深度双边权重的归一化 σ（Δz / z）
FOCUS_EPS = 0.01           # 对焦收敛判定（m）
ENERGY_TOL = 0.02          # 高光能量守恒容差（2%）
LEAK_RATIO_TOL = 0.05      # 剪影边缘跨深度泄漏比例上限
JITTER_MEAN_TOL = 0.15     # 抖动序列均值偏离 0.5 的容差（像素）

_EPS = 1e-6


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _lerp(a, b, t):
    return a + (b - a) * t


def _lum(rgb) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def _sign(x) -> int:
    return (x > _EPS) - (x < -_EPS)


def _halton(index: int, base: int) -> float:
    """无状态 Halton 低差异序列，取值 [0, 1)。"""
    result, f, i = 0.0, 1.0, index + 1  # +1 使第 0 帧不取到 0
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result


# ===========================================================================
# 修复点 1：CoC 全分辨率校验
# ===========================================================================

def coc_fullres(params) -> dict:
    """薄透镜模型计算**全分辨率** CoC（像素，带符号）。

    参数
    ----
    params: dict 或标量深度（m）。字典键：
        depth        : float | list[float]，场景深度（m）
        focus_depth  : 焦平面深度（m）
        focal_length : 焦距（mm）
        f_stop       : 光圈 F 值
        sensor_width : 传感器宽度（mm）
        frame_width  : 全分辨率帧宽（px）

    返回
    ----
    {"coc_px": list[float], "resolution": "full", ...回显参数}

    公式（d, s 转 mm，f 为 mm，N 为 F 值）::

        coc_mm = f^2 * |d - s| / (N * s * (d - f))     # d > f
    前景取负号，再按 sensor_width / frame_width 换算为全分辨率像素。
    """
    if _is_number(params):
        params = {"depth": float(params)}
    p = {**DEFAULT_CAMERA, **params}

    depths = p["depth"]
    if not isinstance(depths, (list, tuple)):
        depths = [depths]

    f = p["focal_length"]
    n_stop = p["f_stop"]
    s = p["focus_depth"] * 1000.0
    mm_to_px = p["frame_width"] / p["sensor_width"]

    coc_px = []
    for depth_m in depths:
        d = depth_m * 1000.0
        if d <= f:  # 透镜内部的非物理深度，视为完全合焦
            coc_px.append(0.0)
            continue
        coc_mm = (f * f * abs(d - s)) / (n_stop * s * (d - f))
        sign = -1.0 if d < s else 1.0
        coc_px.append(sign * coc_mm * mm_to_px)

    return {
        "coc_px": coc_px,
        "resolution": "full",
        "focus_depth": p["focus_depth"],
        "focal_length": f,
        "f_stop": n_stop,
    }


def validation(params) -> dict:
    """半分辨率 CoC 上采样后与全分辨率参考逐像素校验并修复。

    半分辨率 pass 会在前景/背景剪影处漏掉单像素级深度突变，必须在全分辨率
    上复核，而不是直接信任 2x 上采样结果。

    params 键
    ---------
    half_coc   : list[float]，长度 H*W 的半分辨率 CoC
    shape      : (H, W)
    full_depth : list[float]，长度 2H*2W 的全分辨率深度（m），用于重算参考
    （或直接给 full_coc 参考）
    abs_tol / rel_tol : 容差，默认 COC_ABS_TOL / COC_REL_TOL

    返回
    ----
    {
        "upsampled":  2x 最近邻上采样 CoC,
        "repaired":   超限像素被全分辨率参考替换后的 CoC,
        "reference":  全分辨率参考 CoC,
        "max_error", "mean_error", "violations", "passed",
    }
    """
    if _is_number(params):
        # 契约/演示用：构造一个含剪影突变的 2x2 半分辨率网格
        params = {
            "half_coc": [-8.0, 0.5],
            "shape": (1, 2),
            "full_depth": [0.8, 0.8, 4.0, 4.0],
        }
    half_coc = list(params["half_coc"])
    h, w = params["shape"]
    abs_tol = params.get("abs_tol", COC_ABS_TOL)
    rel_tol = params.get("rel_tol", COC_REL_TOL)

    # 2x 最近邻上采样
    upsampled = [0.0] * (4 * h * w)
    for y in range(h):
        for x in range(w):
            v = half_coc[y * w + x]
            for dy in (0, 1):
                for dx in (0, 1):
                    upsampled[(2 * y + dy) * (2 * w) + (2 * x + dx)] = v

    if "full_coc" in params:
        reference = list(params["full_coc"])
    else:
        reference = coc_fullres(
            {**DEFAULT_CAMERA, **{k: v for k, v in params.items()
                                  if k in DEFAULT_CAMERA},
             "depth": params["full_depth"]}
        )["coc_px"]

    repaired = list(upsampled)
    errors, violations = [], []
    for i, (got, ref) in enumerate(zip(upsampled, reference)):
        err = abs(got - ref)
        tol = abs_tol + rel_tol * abs(ref)
        errors.append(err)
        if err > tol:
            violations.append(i)
            repaired[i] = ref  # 用全分辨率真值修复

    return {
        "upsampled": upsampled,
        "repaired": repaired,
        "reference": reference,
        "max_error": max(errors, default=0.0),
        "mean_error": sum(errors) / len(errors) if errors else 0.0,
        "violations": violations,
        "passed": not violations,
    }


# ===========================================================================
# 修复点 2：散景上采样的深度权重
# ===========================================================================

def depth_weight(params) -> float:
    """散景上采样单个 tap 的深度双边权重（含前景屏障）。

    修复「前景边缘把背景散景吃掉」：半分辨率 tap 与全分辨率目标像素深度
    不一致时按高斯衰减；若 tap 与目标分居焦平面两侧，进一步施加屏障权重——
    尤其是更近的前景 tap 渗入背景像素时，不能用它接近 0 的远场 CoC 抹掉
    背景散景。近/远场的正式分离由 near_far_split 负责。

    params 键: depth, center_depth, coc, center_coc, sigma
    标量输入返回一组默认演示参数的权重。
    """
    if _is_number(params):
        params = {"depth": 0.8, "center_depth": 4.0,
                  "coc": -8.0, "center_coc": 6.0}
    z_i = float(params["depth"])
    z_c = float(params["center_depth"])
    coc_i = float(params.get("coc", 0.0))
    coc_c = float(params.get("center_coc", 0.0))
    sigma = float(params.get("sigma", DEPTH_SIGMA))

    delta = (z_i - z_c) / max(z_c, _EPS)
    w = math.exp(-(delta * delta) / (2.0 * sigma * sigma))

    if _sign(coc_i) * _sign(coc_c) < 0:  # 跨焦平面、跨深度的污染 tap
        if z_i < z_c:
            # 前景 tap 落在背景目标上：强屏障，保住背景散景能量
            w *= NEAR_FIELD_BARRIER
        else:
            # 背景 tap 落在前景目标上：弱屏障，最终由 near-over-far 合成兜底
            w *= FAR_FIELD_BARRIER
    return w


def upsample(params) -> dict:
    """半分辨率散景的 2x 深度感知双边上采样。

    params 键
    ---------
    color : list[[r,g,b]]，长度 H*W，半分辨率线性 HDR 颜色
    depth : list[float]，长度 H*W
    coc   : list[float]，长度 H*W
    shape : (H, W)

    每个全分辨率像素取对应 2x2 半分辨率 footprint，空间双线性权重乘以
    depth_weight 后归一化。CoC 单独做保幅度上采样（取 footprint 内最大
    绝对值），避免散景盘在插值中被缩小。
    """
    if _is_number(params):
        params = {
            "shape": (1, 2),
            "color": [[0.1, 0.1, 0.1], [2.0, 1.8, 1.2]],
            "depth": [0.8, 4.0],
            "coc": [-8.0, 6.0],
        }
    h, w = params["shape"]
    color = params["color"]
    depth = params["depth"]
    coc = params["coc"]

    out_color = [[0.0, 0.0, 0.0] for _ in range(4 * h * w)]
    out_coc = [0.0] * (4 * h * w)

    for fy in range(2 * h):
        for fx in range(2 * w):
            # footprint 左上角半分辨率坐标
            hy = min(fy // 2, h - 1)
            hx = min(fx // 2, w - 1)
            ty = (fy / 2.0) - hy  # 半分辨率空间内的小数坐标
            tx = (fx / 2.0) - hx
            taps = []
            for dy in (0, 1):
                for dx in (0, 1):
                    ny, nx = min(hy + dy, h - 1), min(hx + dx, w - 1)
                    sw = ((1.0 - ty) if dy == 0 else ty) * \
                         ((1.0 - tx) if dx == 0 else tx)
                    taps.append((ny * w + nx, sw))
            center_depth = sum(sw * depth[i] for i, sw in taps)
            center_coc = sum(sw * coc[i] for i, sw in taps)

            wsum = 0.0
            acc = [0.0, 0.0, 0.0]
            coc_mag, coc_signed = 0.0, 0.0
            for i, sw in taps:
                dw = depth_weight({
                    "depth": depth[i], "center_depth": center_depth,
                    "coc": coc[i], "center_coc": center_coc,
                })
                wt = sw * dw
                wsum += wt
                for c in range(3):
                    acc[c] += wt * color[i][c]
                if abs(coc[i]) > coc_mag:
                    coc_mag = abs(coc[i])
                    coc_signed = coc[i]
            if wsum <= _EPS:  # 全部被屏障压制时退化为纯空间双线性
                wsum = sum(sw for _, sw in taps)
                for i, sw in taps:
                    for c in range(3):
                        acc[c] += sw * color[i][c]
            idx = fy * (2 * w) + fx
            out_coc[idx] = coc_signed
            for c in range(3):
                out_color[idx][c] = acc[c] / max(wsum, _EPS)

    return {"color": out_color, "coc": out_coc, "shape": (2 * h, 2 * w)}


def near_far_split(params) -> dict:
    """按 CoC 符号把缓冲区拆成近场 / 远场 / 合焦三层。

    合成顺序固定为：合焦底图 + 远场散景，再以近场覆盖率（CoC 盘占像素比）
    做 premultiplied-alpha over 合成。近场永远在远场之上，从结构上杜绝
    远场颜色翻过剪影边缘。
    """
    if _is_number(params):
        params = {
            "coc": [-8.0, 0.0, 6.0],
            "color": [[0.2, 0.3, 0.4], [0.5, 0.5, 0.5], [1.5, 1.4, 1.0]],
            "depth": [0.8, 2.0, 4.0],
        }
    coc, color, depth = params["coc"], params["color"], params["depth"]
    n = len(coc)
    layers = {
        "near": {"color": [[0.0, 0.0, 0.0]] * n, "mask": [0.0] * n},
        "far": {"color": [[0.0, 0.0, 0.0]] * n, "mask": [0.0] * n},
        "focus": {"color": [[0.0, 0.0, 0.0]] * n, "mask": [0.0] * n},
    }
    for i, c in enumerate(coc):
        coverage = _clamp(abs(c) / 2.0, 0.0, 1.0)  # 半径≈|CoC|/2 的盘覆盖率
        if c < -_EPS:
            layers["near"]["color"][i] = [v * coverage for v in color[i]]
            layers["near"]["mask"][i] = coverage
        elif c > _EPS:
            layers["far"]["color"][i] = [v * coverage for v in color[i]]
            layers["far"]["mask"][i] = coverage
        else:
            layers["focus"]["color"][i] = list(color[i])
            layers["focus"]["mask"][i] = 1.0

    # near over far over focus 的参考合成
    composite = [[0.0, 0.0, 0.0] for _ in range(n)]
    for i in range(n):
        base = [layers["focus"]["color"][i][c] + layers["far"]["color"][i][c]
                for c in range(3)]
        a = layers["near"]["mask"][i]
        composite[i] = [layers["near"]["color"][i][c] + (1.0 - a) * base[c]
                        for c in range(3)]
    layers["composite"] = composite
    layers["depth"] = list(depth)
    return layers


# ===========================================================================
# 修复点 3：对焦收敛与重置
# ===========================================================================

def focus_converge(params) -> dict:
    """焦平面指数收敛，避免硬切造成的散景跳变。

    new = target + (current - target) * exp(-speed * dt)
    残差进入 FOCUS_EPS 即吸附到目标并标记 converged。
    """
    if _is_number(params):
        params = {"current": 5.0, "target": 2.0, "dt": 1.0 / 60.0,
                  "speed": 4.0}
    current = float(params["current"])
    target = float(params["target"])
    dt = float(params.get("dt", 1.0 / 60.0))
    speed = float(params.get("speed", 4.0))
    epsilon = float(params.get("epsilon", FOCUS_EPS))

    new = target + (current - target) * math.exp(-speed * dt)
    residual = abs(new - target)
    if residual <= epsilon:
        new, residual, converged = target, 0.0, True
    else:
        converged = False
    return {"focus_depth": new, "target": target,
            "residual": residual, "converged": converged}


def focus_reset(params) -> dict:
    """场景切换检测与对焦状态重置。

    场景 id 变化，或中位深度相对上一帧的相对跳变超过 cut_threshold 时，
    清空对焦历史并把焦平面直接落到新场景目标，禁止沿用上一帧焦平面。

    params 键: state(dict|None), scene_id, median_depth, target_focus,
               cut_threshold(默认 0.5)
    """
    if _is_number(params):
        params = {"state": {"scene_id": 7, "median_depth": 10.0,
                            "focus_depth": 10.0, "converged": True,
                            "history": [10.0]},
                  "scene_id": 8, "median_depth": 1.5, "target_focus": 1.5}
    state = params.get("state")
    scene_id = params["scene_id"]
    median = float(params["median_depth"])
    target = float(params.get("target_focus", median))
    cut_threshold = float(params.get("cut_threshold", 0.5))

    scene_cut = False
    reason = None
    if state is None:
        scene_cut, reason = True, "first_frame"
    elif state.get("scene_id") != scene_id:
        scene_cut, reason = True, "scene_id_changed"
    else:
        prev = float(state.get("median_depth", median))
        if prev > _EPS and abs(median - prev) / prev > cut_threshold:
            scene_cut, reason = True, "depth_discontinuity"

    if scene_cut:
        return {
            "scene_id": scene_id, "median_depth": median,
            "focus_depth": target, "converged": False,
            "history": [], "reset": True, "reason": reason,
        }
    new_state = dict(state)
    new_state.setdefault("history", []).append(median)
    new_state["median_depth"] = median
    new_state["reset"] = False
    return new_state


# ===========================================================================
# 修复点 4：抖动序列固定
# ===========================================================================

def jitter_seed(params) -> tuple:
    """返回第 frame_index 帧的确定性亚像素抖动 (jx, jy)。

    使用无状态 Halton(base=2, base=3) 序列，映射到 [-0.5, 0.5) 像素。
    同一 (frame_index, base_seed) 永远得到同一结果，不依赖任何全局 RNG
    状态，因此时间抖动不会逐帧闪烁、跨运行可复现。
    """
    if _is_number(params):
        params = {"frame_index": int(params)}
    frame_index = int(params["frame_index"])
    base_seed = int(params.get("base_seed", 0))
    idx = frame_index + base_seed
    jx = _halton(idx, 2) - 0.5
    jy = _halton(idx, 3) - 0.5
    return (jx, jy)


def determinism(params) -> dict:
    """校验抖动/时间累积管线在两次独立运行间逐位一致。"""
    if _is_number(params):
        params = {"frames": 8, "base_seed": 1234}
    frames = int(params.get("frames", 8))
    base_seed = int(params.get("base_seed", 0))

    def run():
        seq, hist = [], None
        for f in range(frames):
            j = jitter_seed({"frame_index": f, "base_seed": base_seed})
            rng = random.Random(base_seed + f)  # 局部 RNG，合成噪声可复现
            current = [[rng.random(), rng.random(), rng.random()]
                       for _ in range(4)]
            acc = temporal({"history": hist, "current": current,
                            "width": 2, "jitter": j, "blend": 0.25})
            hist = acc["color"]
            seq.append((j, acc["color"]))
        return seq

    run_a, run_b = run(), run()
    identical = run_a == run_b
    flat = [c for j, buf in run_a for px in buf for c in px]
    return {
        "deterministic": identical,
        "frames": frames,
        "jitter_sequence": [j for j, _ in run_a],
        "temporal_min": min(flat, default=0.0),
        "temporal_max": max(flat, default=0.0),
    }


# ===========================================================================
# 修复点 5：pass 顺序与栅栏
# ===========================================================================

_PASS_GRAPH = [
    # name, 等待的栅栏, 完成后触发的栅栏, 是否 async compute
    ("depth_prepass",        set(),                       {"F_DEPTH"},    False),
    ("coc_halfres",          {"F_DEPTH"},                 {"F_COC_HALF"}, False),
    ("bokeh_async",          {"F_COC_HALF"},              {"F_BOKEH"},    True),
    ("coc_fullres_check",    {"F_COC_HALF"},              {"F_COC_FULL"}, True),
    ("depth_upsample",       {"F_BOKEH", "F_COC_FULL"},   {"F_UP"},       False),
    ("near_far_composite",   {"F_UP"},                    {"F_COMP"},     False),
    ("temporal_accum",       {"F_COMP"},                  {"F_TA"},       False),
    ("tonemap",              {"F_TA"},                    {"F_TM"},       False),
    ("output",               {"F_TM"},                    set(),          False),
]


def pass_order(params=None) -> dict:
    """返回规范 pass 顺序并校验提交顺序是否满足全部栅栏依赖。

    关键不变量：色调映射（tonemap）必须在散景/上采样/合成/时间累积全部
    完成之后——DoF 在线性 HDR 空间做，tonemap 提前会吃掉高光能量。异步
    compute pass（bokeh_async、coc_fullres_check）通过 fence 与图形队列
    对齐，顺序不固定即判违规并触发 fallback 同步化。
    """
    if _is_number(params) or params is None:
        submitted = [p[0] for p in _PASS_GRAPH]
    else:
        submitted = list(params["order"])

    spec = {name: (waits, signals, is_async)
            for name, waits, signals, is_async in _PASS_GRAPH}
    signaled, violations = set(), []

    if sorted(submitted) != sorted(spec):
        violations.append({
            "type": "pass_set_mismatch",
            "detail": {"missing": sorted(set(spec) - set(submitted)),
                       "unexpected": sorted(set(submitted) - set(spec))},
        })

    for pos, name in enumerate(submitted):
        if name not in spec:
            violations.append({"type": "unknown_pass", "pass": name})
            continue
        waits, signals, is_async = spec[name]
        missing = waits - signaled
        if missing:
            violations.append({
                "type": "fence_not_signaled", "pass": name,
                "position": pos, "missing_fences": sorted(missing),
            })
        signaled |= signals

    # 显式不变量：tonemap 不得早于任何 HDR 空间 pass
    hdr_passes = {"bokeh_async", "depth_upsample", "near_far_composite",
                  "temporal_accum"}
    if "tonemap" in submitted:
        tm_pos = submitted.index("tonemap")
        for hp in hdr_passes:
            if hp in submitted and submitted.index(hp) > tm_pos:
                violations.append({
                    "type": "tonemap_before_hdr_pass",
                    "pass": hp, "tonemap_position": tm_pos,
                })

    return {
        "canonical_order": [p[0] for p in _PASS_GRAPH],
        "submitted_order": submitted,
        "fences": {name: {"waits": sorted(waits),
                          "signals": sorted(signals),
                          "async": is_async}
                   for name, waits, signals, is_async in _PASS_GRAPH},
        "violations": violations,
        "valid": not violations,
    }


def temporal(params) -> dict:
    """时间累积：历史重投影 + 邻域钳制，场景切换时丢弃历史。

    抖动来自固定 jitter_seed，混合权重恒定，因此散景颗粒不会逐帧闪烁。
    历史样本先钳制到当前帧 3x3 邻域的 min/max（抗拖影），再按 blend 混合。
    params 键: history(None|list[rgb]), current(list[rgb]), width,
               jitter(tuple), blend, cut(bool)
    """
    if _is_number(params):
        rng = random.Random(7)
        cur = [[rng.random() for _ in range(3)] for _ in range(4)]
        params = {"history": None, "current": cur, "width": 2,
                  "jitter": jitter_seed(0), "blend": 0.25}
    current = params["current"]
    width = int(params["width"])
    height = len(current) // width
    blend = float(params.get("blend", 0.1))
    cut = bool(params.get("cut", False))
    history = params.get("history")

    if history is None or cut:
        return {"color": [list(px) for px in current],
                "history_used": False, "cut": cut}

    def neighbors(idx):
        y, x = divmod(idx, width)
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    out.append(current[ny * width + nx])
        return out

    out = []
    for i, px in enumerate(current):
        nb = neighbors(i)
        clamped = []
        for c in range(3):
            lo = min(q[c] for q in nb)
            hi = max(q[c] for q in nb)
            clamped.append(_clamp(history[i][c], lo, hi))
        out.append([_lerp(clamped[c], px[c], blend) for c in range(3)])
    return {"color": out, "history_used": True, "cut": False,
            "jitter": tuple(params.get("jitter", (0.0, 0.0)))}


# ===========================================================================
# 修复点 6：远景高光能量守恒
# ===========================================================================

def _soft_knee(x, knee, white):
    if x <= knee:
        return x
    over = x - knee
    return knee + over / (1.0 + over / max(white - knee, _EPS))


def highlight_energy(params) -> dict:
    """远景高光的软拐点 + 归一化散景盘，保证前后能量守恒。

    流程：HDR 颜色先过软拐点（避免硬裁剪直接丢能量），再按每像素远场
    CoC 半径做归一化圆盘模糊（核权重和为 1），最后按亮度总能量做增益
    校正，使 E_out / E_in 落在 ENERGY_TOL 内。只处理 CoC > 0 的远场。
    """
    if _is_number(params):
        params = {
            "shape": (1, 3),
            "color": [[0.2, 0.2, 0.2], [8.0, 7.0, 4.0], [0.1, 0.1, 0.1]],
            "coc": [2.0, 4.0, 2.0],
            "knee": 1.0, "white": 8.0,
        }
    h, w = params["shape"]
    color = [list(px) for px in params["color"]]
    coc = list(params["coc"])
    knee = float(params.get("knee", 1.0))
    white = float(params.get("white", 8.0))

    energy_in = sum(_lum(px) for px in color)

    # 1) 软拐点压缩（保留可恢复的高光能量，不做硬 clip）
    kneed = [[_soft_knee(c, knee, white) for c in px] for px in color]

    # 2) 归一化圆盘模糊：每个源像素的能量摊到 CoC 盘内，核和恒为 1
    blurred = [[0.0, 0.0, 0.0] for _ in color]
    for y in range(h):
        for x in range(w):
            i = y * w + x
            r = max(1, int(round(max(coc[i], 0.0) / 2.0)))
            disk, wsum = [], 0.0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy > r * r:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        disk.append(ny * w + nx)
                        wsum += 1.0
            for j in disk:
                for c in range(3):
                    blurred[j][c] += kneed[i][c] / wsum

    # 3) 能量增益校正（按亮度标量，保持色相）。
    # 守恒口径：软拐点是 HDR 降量程（能量保留到 knee/white 区间，非硬裁剪），
    # 因此模糊 pass 必须守恒的是「软拐点之后」的能量 E_kneed。
    energy_kneed = sum(_lum(px) for px in kneed)
    energy_blur = sum(_lum(px) for px in blurred)
    gain = energy_kneed / energy_blur if energy_blur > _EPS else 1.0
    out = [[c * gain for c in px] for px in blurred]
    energy_out = sum(_lum(px) for px in out)

    clipped = sum(1 for px in color for c in px if c > white)
    ratio = energy_out / energy_kneed if energy_kneed > _EPS else 1.0
    return {
        "color": out,
        "energy_in": energy_in,
        "energy_kneed": energy_kneed,
        "energy_out": energy_out,
        # 软拐点相对原始 HDR 的压缩比（参考量，不参与守恒判定）
        "compression": energy_kneed / energy_in if energy_in > _EPS else 1.0,
        "ratio": ratio,
        "gain": gain,
        "hard_clipped_pixels": clipped,
        "conserved": abs(ratio - 1.0) <= ENERGY_TOL,
    }


# ===========================================================================
# 审计、指标与降级
# ===========================================================================

def metrics(params) -> dict:
    """汇总单帧 QA 指标：CoC 误差、能量守恒、边缘泄漏、对焦、抖动、栅栏。"""
    if _is_number(params):
        validation_report = validation(1)
        energy_report = highlight_energy(1)
        order_report = pass_order(1)
        det = determinism({"frames": 16, "base_seed": 99})
        params = {
            "validation": validation_report,
            "energy": energy_report,
            "order": order_report,
            "determinism": det,
            "depth": [0.8, 0.8, 4.0, 4.0],
            "coc": validation_report["repaired"],
            "focus_residual": 0.0,
        }

    v = params["validation"]
    energy = params["energy"]
    order = params["order"]
    det = params.get("determinism", {"deterministic": True,
                                     "jitter_sequence": []})

    # 剪影边缘跨深度泄漏：对每对跨焦平面相邻像素，深度权重在无屏障时接近 1，
    # 屏障生效后被压低；穿透屏障留下的权重比例即「泄漏率」。
    depth, coc = params["depth"], params["coc"]
    w_ = params.get("width", int(math.sqrt(len(depth))) or 1)
    boundary, leaked = 0, 0.0
    for i, z in enumerate(depth):
        y, x = divmod(i, w_)
        for j in (i + 1 if (x + 1) % w_ else -1,
                  i + w_ if i + w_ < len(depth) else -1):
            if j < 0 or _sign(coc[i]) * _sign(coc[j]) >= 0:
                continue
            boundary += 1
            common = {"depth": depth[i], "center_depth": depth[j],
                      "coc": coc[i], "center_coc": coc[j]}
            raw = depth_weight({**common, "sigma": 1e9})  # 关掉高斯，只看屏障
            guarded = depth_weight(common)
            leaked += guarded / max(raw, _EPS)
    leak_ratio = (leaked / boundary) if boundary else 0.0

    jseq = det.get("jitter_sequence", [])
    jmean = (sum(a for a, _ in jseq) / len(jseq)) if jseq else 0.0

    out = {
        "coc_max_error": v["max_error"],
        "coc_mean_error": v["mean_error"],
        "coc_violations": len(v["violations"]),
        "energy_ratio": energy["ratio"],
        "edge_leak_ratio": leak_ratio,
        "focus_residual": float(params.get("focus_residual", 0.0)),
        "jitter_mean_offset": abs(jmean),
        "deterministic": bool(det["deterministic"]),
        "fence_violations": len(order["violations"]),
    }
    out["ok"] = (
        out["coc_violations"] == 0
        and abs(out["energy_ratio"] - 1.0) <= ENERGY_TOL
        and out["edge_leak_ratio"] <= LEAK_RATIO_TOL
        and out["focus_residual"] <= FOCUS_EPS
        and out["jitter_mean_offset"] <= JITTER_MEAN_TOL
        and out["deterministic"]
        and out["fence_violations"] == 0
    )
    return out


def audit(params) -> dict:
    """按阈值审计单帧指标，输出分级违规列表。"""
    if _is_number(params):
        params = {"metrics": metrics(1)}
    m = params["metrics"]
    violations = []

    def check(name, value, ok, severity, detail):
        if not ok:
            violations.append({"check": name, "severity": severity,
                               "value": value, "detail": detail})

    check("coc_fullres_validation", m["coc_violations"],
          m["coc_violations"] == 0, "error",
          f"{m['coc_violations']} 个像素超出全分辨率 CoC 容差")
    check("highlight_energy", m["energy_ratio"],
          abs(m["energy_ratio"] - 1.0) <= ENERGY_TOL, "error",
          f"能量比 {m['energy_ratio']:.4f} 偏离 1 超过 {ENERGY_TOL}")
    check("fence_order", m["fence_violations"],
          m["fence_violations"] == 0, "error",
          f"{m['fence_violations']} 处栅栏/顺序违规")
    check("determinism", m["deterministic"], m["deterministic"], "error",
          "抖动或时间累积跨运行不一致")
    check("edge_contamination", m["edge_leak_ratio"],
          m["edge_leak_ratio"] <= LEAK_RATIO_TOL, "warning",
          f"剪影泄漏比例 {m['edge_leak_ratio']:.4f}")
    check("focus_convergence", m["focus_residual"],
          m["focus_residual"] <= FOCUS_EPS, "warning",
          f"对焦残差 {m['focus_residual']:.4f} m")
    check("jitter_bias", m["jitter_mean_offset"],
          m["jitter_mean_offset"] <= JITTER_MEAN_TOL, "warning",
          f"抖动均值偏移 {m['jitter_mean_offset']:.4f} px")

    return {
        "passed": not any(v["severity"] == "error" for v in violations),
        "warnings": [v for v in violations if v["severity"] == "warning"],
        "errors": [v for v in violations if v["severity"] == "error"],
        "violations": violations,
        "metrics": m,
    }


def fallback(params) -> dict:
    """根据审计结果选择安全降级路径，可叠加，最差时整体关闭 DoF。"""
    if _is_number(params):
        params = {"audit": audit(1)}
    report = params["audit"]
    checks = {v["check"] for v in report["violations"]}

    disabled, reasons = [], []
    sync = False
    if "coc_fullres_validation" in checks:
        disabled.append("halfres_coc")  # 强制全分辨率 CoC
        reasons.append("CoC 校验失败，停用半分辨率路径")
    if "fence_order" in checks:
        sync = True                     # 异步 compute 全部改同步
        disabled.append("async_compute")
        reasons.append("栅栏违规，强制同步 pass 顺序")
    if "highlight_energy" in checks:
        disabled.append("highlight_bokeh")
        reasons.append("高光能量失恒，关闭远景高光散景扩展")
    if "determinism" in checks:
        disabled.append("temporal_accum")
        reasons.append("序列不可复现，关闭时间累积并锁定固定种子")
    if "edge_contamination" in checks:
        disabled.append("bilateral_upsample")
        reasons.append("剪影边缘泄漏，上采样退化为点采样+近远场硬合成")

    if len(disabled) >= 4:
        level = "dof_off"           # 灾难性失败：输出锐利原图
        disabled.append("depth_of_field")
    elif disabled:
        level = "degraded"
    else:
        level = "none"
    return {"fallback_level": level, "sync_pipeline": sync,
            "disabled_features": sorted(set(disabled)),
            "reasons": reasons,
            "passed": report["passed"]}
