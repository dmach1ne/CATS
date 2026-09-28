# Third-party review

Attribution inventory, not an approved blanket license. Original source headers
and public source URLs are retained.

- Benchmark code comes from the AFQR research checkout and includes modified
  Stable-Baselines3 implementations. The installed SB3 2.5.0 MIT license is
  preserved under licenses/.
- CAPS, L2C2, Grad-CAPS, ASAP and LipsNet++ comparison implementations still need
  an exact upstream-version and redistribution-notice review. A Git remote alone
  is not license evidence.
- er_isaac simulation files retain Isaac Lab BSD-3-Clause SPDX/copyright headers.
  Archived setup metadata instead said Apache-2.0; this conflict is unresolved.
  That setup.py is not distributed or used by the local-path launcher.
- Isaac Sim/Lab, rl_games, Gymnasium, MuJoCo, PyTorch and assets remain separately
  installed dependencies under their maintainers' terms, not vendored binaries.

Before upload, resolve notices/permissions for copied and derived code, then add
the owner's chosen license for original contributions. Until then this is private
staging, not licensed public code.
