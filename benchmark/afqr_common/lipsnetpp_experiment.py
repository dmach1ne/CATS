"""Shared LipsNet++ experiment profiles and artifact naming."""

from dataclasses import asdict, dataclass
from afqr_common.experiment_settings import selected_settings


@dataclass(frozen=True)
class LipsNetPlusPlusProfile:
    name: str
    lambda_k: float
    lambda_h: float
    history_len: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def selected_profile(algorithm: str, environment: str) -> LipsNetPlusPlusProfile:
    settings = selected_settings(algorithm, environment)["lipsnetpp"]
    return LipsNetPlusPlusProfile(
        settings["profile_name"], settings["lipsnetpp_lambda_k"],
        settings["lipsnetpp_lambda_h"], settings["history_len"],
    )


def scalar_label(value: float) -> str:
    return f"{value:g}"


def run_name(algorithm: str, profile: LipsNetPlusPlusProfile, seed: int) -> str:
    return (
        f"lipsnetpp_{algorithm.lower()}"
        f"_lk{scalar_label(profile.lambda_k)}"
        f"_lh{scalar_label(profile.lambda_h)}"
        f"_N{profile.history_len}_{seed}"
    )
