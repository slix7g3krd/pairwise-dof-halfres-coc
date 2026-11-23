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
