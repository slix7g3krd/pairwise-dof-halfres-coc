# pairwise-dof-halfres-coc

## 领域

游戏、图形与多媒体

## 主题

半分辨率景深

## 说明

CoC 校验、对焦收敛与抖动序列固定。

初始状态下 `graphics/dof_halfres_coc.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

## 修复点与函数对照

| 修复点 | 函数 |
| --- | --- |
| CoC 全分辨率校验 | `coc_fullres`（薄透镜全分辨率带符号 CoC）、`validation`（半分辨率 2x 上采样后逐像素比对、超限即按全分辨率参考修复） |
| 散景上采样的深度权重 | `depth_weight`（深度双边高斯 + 跨焦平面屏障，前景 tap 渗入背景权重压到 0.05）、`upsample`（深度感知双边上采样，CoC 保幅度）、`near_far_split`（近/远/合焦三层，near-over-far premultiplied-alpha 合成） |
| 对焦收敛与重置 | `focus_converge`（指数收敛 + 残差吸附）、`focus_reset`（场景 id 变化或中位深度突变即清历史，禁止沿用上一帧焦平面） |
| 抖动序列固定 | `jitter_seed`（无状态 Halton(2,3) 亚像素抖动）、`determinism`（两次独立运行逐位一致校验，只用局部 `random.Random(seed)`） |
| pass 顺序与栅栏 | `pass_order`（9 个 pass 的规范序、fence 依赖图与 tonemap-before-HDR 不变量）、`temporal`（固定抖动的历史重投影 + 3x3 邻域钳制 + 切场丢历史）、`fallback`（栅栏违规时强制同步管线） |
| 远景高光能量守恒 | `highlight_energy`（软拐点替代硬裁剪 + 归一化圆盘模糊 + 按亮度的能量增益校正，守恒口径为软拐点之后）、`metrics` / `audit`（CoC 误差、能量比、剪影泄漏率、对焦残差、抖动偏置、确定性、栅栏违规的分级审计与降级） |

### 约定

- 深度对外单位为米；CoC 单位为全分辨率像素，负值=前景、正值=背景、0=合焦。
- 纯标准库实现，无第三方依赖；所有随机量来自局部种子或无状态 Halton 序列，禁止全局 RNG。
- 每个函数接受真实参数字典；传标量（如契约测试的 `1`）时使用内置演示参数。
