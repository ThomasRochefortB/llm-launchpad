"""Where a quantization sits on the quality axis, separate from price.

Fast Deploy ranks placements by price and speed. A smaller quantization wins
both -- fewer bytes per weight is cheaper to hold and faster to read -- so a
ranker that sees only those two axes reaches for the smallest one every time.
That is how all three tiers ended up on 2-bit weights while the model's
published benchmark score, measured at full precision, sat on screen beside
them.

Bit width is the missing third axis. It is not another point on the
price/speed frontier; it is the floor that frontier is explored within.
"""

from __future__ import annotations

import re

# Four bits per weight is the knee of the curve: below it, loss stops being a
# rounding error and starts changing answers. Four bits is emphatically not
# "lossless" -- that is Q8_0 at best and BF16 in truth -- but it is the lowest
# width Fast Deploy will pick on someone's behalf without saying so out loud.
QUALITY_FLOOR_BITS = 4

# Labels whose bit width their own spelling does not give away.
_EXPLICIT_BITS = {
    "MXFP4": 4,
    "MXFP4_MOE": 4,
    "F16": 16,
    "FP16": 16,
    "BF16": 16,
    "F32": 32,
    "FP32": 32,
}
_QUANT_BITS_RE = re.compile(r"(?i)(?:^|[-_])I?Q(\d)")


def quant_bits(quant: str | None) -> int | None:
    """Approximate bits per weight for a GGUF quantization label.

    ``None`` means the label was not recognised, which is a different fact
    from "this is a small quantization" and is kept distinct from it.
    """

    if not quant:
        return None
    label = quant.strip().upper()
    explicit = _EXPLICIT_BITS.get(label.removeprefix("UD-").removeprefix("UD_"))
    if explicit is not None:
        return explicit
    match = _QUANT_BITS_RE.search(label)
    return int(match.group(1)) if match else None


def serving_quality_bits(quant: str | None) -> int:
    """Bit width for ranking, with an unrecognised label treated as adequate.

    An unknown label is not evidence of degradation. Scoring it as zero would
    rank it below a 2-bit quantization and quietly bury whatever it is.
    """

    bits = quant_bits(quant)
    return QUALITY_FLOOR_BITS if bits is None else bits


def is_reduced_quality(quant: str | None) -> bool:
    """Whether serving this quantization visibly degrades the model."""

    bits = quant_bits(quant)
    return bits is not None and bits < QUALITY_FLOOR_BITS


def quant_quality_label(quant: str | None) -> str:
    """Name the bit width, for a caller that has to disclose it."""

    bits = quant_bits(quant)
    return f"{bits}-bit" if bits is not None else ""
