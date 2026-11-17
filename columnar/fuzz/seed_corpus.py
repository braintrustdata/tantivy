from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

def seed(name, shapes, *, kind=0, pattern=1, keep=0, large=False, size=0, cardinality=0):
    data = bytearray(40)
    data[0] = 4
    data[1] = 31 if large else 0
    data[2] = kind
    data[3] = pattern
    data[4:8] = bytes([size if large else 128, 0, 128, 128])
    data[8:20] = bytes(shapes * 4)
    data[23] = keep
    data[24] = cardinality
    data[31] = 4
    data[32:40] = (0x123456789abcdef0).to_bytes(8, "little")
    (args.output / name).write_bytes(data)

for kind in range(5):
    for pattern in range(4):
        seed(f"sparse-{kind}-{pattern}", [3, 2, 1], kind=kind, pattern=pattern)
for shape in range(8):
    seed(f"shape-{shape}", [shape] * 3, cardinality=shape % 3, keep=shape % 8)
for size in range(4):
    for shape in [1, 2, 3, 6]:
        seed(f"block-{size}-{shape}", [shape] * 3, large=True, size=size, cardinality=size % 3, pattern=2)
seed("zero-survivors", [3] * 3, keep=1)
seed("one-survivor", [3] * 3, keep=5)
seed("multivalued", [5] * 3)
seed("full", [4] * 3)
(args.output / "zero-segments").write_bytes(bytes(40))
print(f"Wrote {len(list(args.output.iterdir()))} seed inputs")
