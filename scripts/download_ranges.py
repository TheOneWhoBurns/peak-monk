"""Inclusive download intervals; resume only bytes recorded after successful writes."""
def missing_ranges(size, completed, chunk):
    merged = []
    for a, b in sorted(completed):
        if not 0 <= a <= b < size:
            raise ValueError('Resume interval outside artifact')
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    cursor = 0
    for a, b in [*merged, [size, size]]:
        for start in range(cursor, a, chunk):
            yield start, min(start + chunk - 1, a - 1)
        cursor = b + 1
