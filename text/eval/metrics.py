"""Exact count-based C and S; undefined denominators stay undefined."""
from fractions import Fraction


def c_value(n0, nu, n):
    if any(type(x) is not int for x in (n0, nu, n)) or n <= 0 or not 0 <= n0 <= n or not 0 <= nu <= n:
        raise ValueError('invalid aggregate counts')
    return None if 4 * n0 <= n else Fraction(4 * (n0 - nu), 4 * n0 - n)


def encode(value):
    return None if value is None else {'exact': [value.numerator, value.denominator], 'percent': float(100 * value)}


def metrics(counts, kind, threshold=65):
    cs = {scene: c_value(v['base'], v['cleared'], v['n']) for scene, v in counts.items()}
    out = {'C': {k: encode(v) for k, v in cs.items()}, 'clamped': False}
    if kind == 'S':
        if set(cs) != {'original', 'syntax', 'paraphrase'}:
            raise ValueError('all S scenes required')
        reason = ('scene baseline at or below chance' if any(v is None for v in cs.values()) else
                  'original C must be positive' if cs['original'] <= 0 else None)
        s = None if reason else 100 * min(cs['syntax'] / cs['original'], cs['paraphrase'] / cs['original'])
        out.update(S_percent=None if s is None else float(s), S_exact=None if s is None else [s.numerator, s.denominator],
                   threshold=threshold, passed=None if s is None else s >= Fraction(str(threshold)), unavailable_reason=reason)
    return out
