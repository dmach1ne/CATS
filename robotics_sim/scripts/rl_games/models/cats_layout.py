from __future__ import annotations


def resolve_action_layout(cats_cfg: dict, actions_num: int) -> tuple[int, int | None]:
    """Resolve the arm dimensions and optional non-regularized gripper index."""
    arm_dims = int(cats_cfg.get("arm_dims", actions_num))
    raw_gripper_index = cats_cfg.get("gripper_action_index", None)
    gripper_index = None if raw_gripper_index is None else int(raw_gripper_index)

    if arm_dims < 1 or arm_dims > actions_num:
        raise ValueError(f"CATS arm_dims must be in [1, {actions_num}], got {arm_dims}")
    if gripper_index is not None:
        if not 0 <= gripper_index < actions_num:
            raise ValueError(
                f"CATS gripper_action_index must be in [0, {actions_num}), got {gripper_index}"
            )
        if gripper_index < arm_dims:
            raise ValueError(
                "CATS gripper_action_index must be outside the regularized arm action prefix"
            )
    return arm_dims, gripper_index
