# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class VerifyAdaptiveConfig:
    """Config for the verifier adaptive step-length controller.

    ``query_len = 1 (anchor) + draft_len``.

    Passed as a JSON **string** via ``--adaptive-verify-config '{...}'``
    (or the ``speculative_adaptive_verify_config`` key inside
    ``--speculative-config``), via :meth:`from_json_str` / :meth:`from_dict`.
    Unknown keys are silently ignored.  See ``verify_adaptive.md``.

    The feature is enabled simply by providing the config; there is no
    separate ``enabled`` flag.
    """

    # -----------------------------------------------------------------------
    # Batch-size axis
    # -----------------------------------------------------------------------

    warmup_batch_sizes: Optional[list[int]] = None
    """Explicit batch-size levels to profile.  ``None`` (default) → use the
    engine's full CUDA-graph capture-size list, optionally clamped by
    *min_warmup_batch_size* / *max_warmup_batch_size*."""

    min_warmup_batch_size: Optional[int] = None
    """When *warmup_batch_sizes* is ``None``, drop CUDA-graph sizes below
    this threshold.  ``None`` → no lower bound."""

    max_warmup_batch_size: Optional[int] = None
    """When *warmup_batch_sizes* is ``None``, drop CUDA-graph sizes above
    this threshold.  ``None`` → no upper bound."""

    # -----------------------------------------------------------------------
    # Query-length axis
    # -----------------------------------------------------------------------

    query_len_list: Optional[list[int]] = None
    """Explicit list of per-request query-length levels to profile.
    When set, *query_len_step_per_req* / *min_query_len_per_req* /
    *max_query_len_per_req* are ignored for list generation (all three still
    apply for validation).  Values must be ≥ 1 and ≤
    *num_speculative_tokens* + 1."""

    query_len_step_per_req: int = 2
    """Step between auto-generated query-length levels.  Level 1 (anchor
    only) is always included.  Ignored when *query_len_list* is set."""

    max_query_len_per_req: Optional[int] = None
    """Upper bound for auto-generated query-length levels.
    ``None`` → *num_speculative_tokens* + 1."""

    min_query_len_per_req: int = 1
    """Lower bound for the stepped part of the auto-generated list.
    Level 1 is always included regardless of this value.  Must be ≥ 1."""

    # -----------------------------------------------------------------------
    # Measurement / profiling
    # -----------------------------------------------------------------------

    warmup_seq_lens: int = 512
    """Simulated KV-context length used during cost profiling.  A larger
    value makes attention cost closer to real long-context inference."""

    n_warmup_iters: int = 3
    """Forward passes discarded before timing begins."""

    n_measure_iters: int = 10
    """Forward passes averaged to produce the ITL estimate."""

    # -----------------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------------

    @classmethod
    def from_json_str(cls, s: str) -> "VerifyAdaptiveConfig":
        """Parse from a JSON **string** (not a file path)."""
        try:
            d = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"--adaptive-verify-config: invalid JSON string: {exc}"
            ) from exc
        if not isinstance(d, dict):
            raise ValueError(
                "--adaptive-verify-config: JSON must be an object, got "
                f"{type(d).__name__}"
            )
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "VerifyAdaptiveConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def validate(self, num_speculative_tokens: int) -> None:
        """Raise ``ValueError`` if any field combination is invalid.

        Args:
            num_speculative_tokens: ``num_speculative_tokens`` from the
                active ``SpeculativeConfig``.  Used to derive the effective
                ``max_query_len_per_req`` when the field is ``None``.
        """
        eff_max_q: int = (
            self.max_query_len_per_req
            if self.max_query_len_per_req is not None
            else num_speculative_tokens + 1
        )

        # --- query-length fields -------------------------------------------
        if self.query_len_step_per_req < 1:
            raise ValueError("query_len_step_per_req must be >= 1.")
        if self.min_query_len_per_req < 1:
            raise ValueError("min_query_len_per_req must be >= 1.")
        if self.max_query_len_per_req is not None:
            if self.max_query_len_per_req < 2:
                raise ValueError("max_query_len_per_req must be >= 2.")
            if self.max_query_len_per_req > num_speculative_tokens + 1:
                raise ValueError(
                    f"max_query_len_per_req ({self.max_query_len_per_req}) "
                    f"> num_speculative_tokens + 1 ({num_speculative_tokens + 1})."
                )
        if self.min_query_len_per_req > eff_max_q:
            raise ValueError(
                f"min_query_len_per_req ({self.min_query_len_per_req}) "
                f"> effective max_query_len_per_req ({eff_max_q})."
            )
        if self.query_len_list is not None:
            if len(self.query_len_list) == 0:
                raise ValueError("query_len_list must not be empty.")
            bad = [q for q in self.query_len_list if q < 1]
            if bad:
                raise ValueError(
                    f"All query_len_list entries must be >= 1; got {bad}."
                )
            bad = [q for q in self.query_len_list if q > eff_max_q]
            if bad:
                raise ValueError(
                    f"query_len_list entries {bad} exceed effective "
                    f"max_query_len_per_req ({eff_max_q})."
                )

        # --- batch-size fields ---------------------------------------------
        if self.warmup_batch_sizes is not None:
            if len(self.warmup_batch_sizes) == 0:
                raise ValueError("warmup_batch_sizes must not be empty when set.")
            bad = [bs for bs in self.warmup_batch_sizes if bs < 1]
            if bad:
                raise ValueError(
                    f"All warmup_batch_sizes entries must be >= 1; got {bad}."
                )
        if self.min_warmup_batch_size is not None:
            if self.min_warmup_batch_size < 1:
                raise ValueError("min_warmup_batch_size must be >= 1.")
        if self.max_warmup_batch_size is not None:
            if self.max_warmup_batch_size < 1:
                raise ValueError("max_warmup_batch_size must be >= 1.")
        if (
            self.min_warmup_batch_size is not None
            and self.max_warmup_batch_size is not None
            and self.min_warmup_batch_size > self.max_warmup_batch_size
        ):
            raise ValueError(
                f"min_warmup_batch_size ({self.min_warmup_batch_size}) "
                f"> max_warmup_batch_size ({self.max_warmup_batch_size})."
            )

        # --- measurement fields --------------------------------------------
        if self.warmup_seq_lens < 1:
            raise ValueError("warmup_seq_lens must be >= 1.")
        if self.n_warmup_iters < 0:
            raise ValueError("n_warmup_iters must be >= 0.")
        if self.n_measure_iters < 1:
            raise ValueError("n_measure_iters must be >= 1.")
