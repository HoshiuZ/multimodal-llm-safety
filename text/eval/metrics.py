"""Exact count-based C and S; keep raw S and report its 100% cap separately."""
from fractions import Fraction


def c_value(n0, nu, n):
    if any(type(x) is not int for x in (n0, nu, n)) or n <= 0 or not 0 <= n0 <= n or not 0 <= nu <= n:
        raise ValueError('invalid aggregate counts')
    return None if 4 * n0 <= n else Fraction(4 * (n0 - nu), 4 * n0 - n)


def encode(value):
    """Encode a ratio as an exact fraction and percentage; None stays N/A."""
    return None if value is None else {'exact': [value.numerator, value.denominator], 'percent': float(100 * value)}


def encode_percent(value):
    """Encode a value already expressed in percent units."""
    return None if value is None else {'exact': [value.numerator, value.denominator], 'percent': float(value)}


def s_metrics(scene_c, threshold=65):
    """Return one shared raw/capped S structure for generic and sparse audits."""
    if set(scene_c) != {'original', 'syntax', 'paraphrase'}:
        raise ValueError('all S scenes required')
    reason = ('scene baseline at or below chance' if any(v is None for v in scene_c.values()) else
              'original C must be positive' if scene_c['original'] <= 0 else None)
    if reason:
        raw = capped = None
    else:
        raw = 100 * min(scene_c['syntax'] / scene_c['original'],
                        scene_c['paraphrase'] / scene_c['original'])
        capped = min(raw, Fraction(100))
    passed = None if capped is None else capped >= Fraction(str(threshold))
    return {
        'raw_S': encode_percent(raw),
        'capped_S': encode_percent(capped),
        # Compatibility aliases retain the historical meaning: these are raw S.
        'exact': None if raw is None else [raw.numerator, raw.denominator],
        'percent': None if raw is None else float(raw),
        'threshold': threshold,
        'threshold_percent': threshold,
        'passed': passed,
        'passes_threshold': passed,
        'unavailable_reason': reason,
        'clamped': raw is not None and raw != capped,
        'cap_formula': 'capped_S=min(raw_S,100)',
    }


def metrics(counts, kind, threshold=65):
    scene_c = {scene: c_value(v['base'], v['cleared'], v['n']) for scene, v in counts.items()}
    out = {'C': {k: encode(v) for k, v in scene_c.items()}, 'clamped': False}
    if kind == 'S':
        s = s_metrics(scene_c, threshold)
        out.update(S=s, S_percent=s['percent'], S_exact=s['exact'], threshold=threshold,
                   passed=s['passed'], unavailable_reason=s['unavailable_reason'])
    return out
