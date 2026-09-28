"""Shared LipsNet++ experiment profiles and artifact naming."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LipsNetPlusPlusProfile:
    name: str
    lambda_k: float
    lambda_h: float
    history_len: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


LIPSNETPP_PROFILES = {
    "A": LipsNetPlusPlusProfile("A", lambda_k=1e-2, lambda_h=1e-2, history_len=5),
    "B": LipsNetPlusPlusProfile("B", lambda_k=1e-2, lambda_h=1e-3, history_len=5),
    "C": LipsNetPlusPlusProfile("C", lambda_k=1e-3, lambda_h=1e-3, history_len=5),
    "D": LipsNetPlusPlusProfile("D", lambda_k=1e-2, lambda_h=1e-3, history_len=10),
    "E": LipsNetPlusPlusProfile("E", lambda_k=1e-3, lambda_h=1e-2, history_len=5),
    "F": LipsNetPlusPlusProfile("F", lambda_k=1e-2, lambda_h=1e-2, history_len=10),
    "G": LipsNetPlusPlusProfile("G", lambda_k=1e-3, lambda_h=1e-2, history_len=10),
    "H": LipsNetPlusPlusProfile("H", lambda_k=1e-3, lambda_h=1e-3, history_len=10),
}


def scalar_label(value: float) -> str:
    return f"{value:g}"


def run_name(algorithm: str, profile: LipsNetPlusPlusProfile, seed: int) -> str:
    return (
        f"lipsnetpp_{algorithm.lower()}"
        f"_lk{scalar_label(profile.lambda_k)}"
        f"_lh{scalar_label(profile.lambda_h)}"
        f"_N{profile.history_len}_{seed}"
    )
