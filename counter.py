from collections import Counter, defaultdict

# ─────────────────────────────────────────
# 1. Basic Counter usage
# ─────────────────────────────────────────
detected = ["cola", "cola", "water", "juice", "cola", "water"]

counts = Counter(detected)
print(counts)
# Counter({'cola': 3, 'water': 2, 'juice': 1})

print(counts["cola"])   # 3
print(counts["beer"])   # 0  (missing keys return 0, no KeyError)

# Most common items
print(counts.most_common(2))
# [('cola', 3), ('water', 2)]

# ─────────────────────────────────────────
# 2. Incrementing counts (like service.py does)
# ─────────────────────────────────────────
bottle_counts = Counter()

# Simulate adding detections one by one
bottle_counts["cola"] += 3
bottle_counts["water"] += 2
bottle_counts["juice"] += 1

print(bottle_counts)
# Counter({'cola': 3, 'water': 2, 'juice': 1})

# ─────────────────────────────────────────
# 3. defaultdict(int) — the other pattern used in service.py
# ─────────────────────────────────────────
bottle_counts_dd = defaultdict(int)   # missing keys default to 0

bottle_counts_dd["cola"] += 3        # works without pre-initializing
bottle_counts_dd["water"] += 2

# Convert to plain dict — what `counts = dict(bottle_counts)` does
counts = dict(bottle_counts_dd)
print(counts)
# {'cola': 3, 'water': 2}

# Why convert?  defaultdict silently creates keys on access:
#   bottle_counts_dd["nonexistent"]  →  adds "nonexistent": 0
# A plain dict raises KeyError instead, which is safer to pass around.

# ─────────────────────────────────────────
# 4. Counter vs defaultdict(int)
# ─────────────────────────────────────────
# Counter     — best when counting from a list/iterable at once
# defaultdict — best when accumulating counts in a loop (like service.py cap→bottle matching)

names = ["cola", "cola", "water"]

from_list  = Counter(names)               # one-shot from list
from_loop  = defaultdict(int)
for n in names:
    from_loop[n] += 1

print(dict(from_list) == dict(from_loop))  # True — same result

# ─────────────────────────────────────────
# 5. Merging / arithmetic
# ─────────────────────────────────────────
a = Counter(cola=3, water=2)
b = Counter(cola=1, juice=5)

print(a + b)   # Counter({'juice': 5, 'cola': 4, 'water': 2})
print(a - b)   # Counter({'water': 2, 'cola': 2})  — drops negatives
