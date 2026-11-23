import pytest
import graphics.dof_halfres_coc as mod

CASES = [
    ("coc_fullres", (1)),
    ("depth_weight", (1)),
    ("focus_converge", (1)),
    ("focus_reset", (1)),
    ("jitter_seed", (1)),
    ("pass_order", (1)),
    ("highlight_energy", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("upsample", (1)),
    ("metrics", (1)),
    ("near_far_split", (1)),
    ("temporal", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
