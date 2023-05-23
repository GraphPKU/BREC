# Non-GNNs Reproduction

## Requirements

In addition to general results, [pynauty](https://pypi.org/project/pynauty/) is also needed.

## Usages

To reproduce best result on 3-WL, run (we recommend to use 2-FWL with same expressiveness but less complexity):

```bash
python test.py --wl 2 --method fwl
```

or

```bash
python test.py --wl 3 --method k-wl
```

To reproduce best result on $S_3$, $S_4$, $N_1$, $N_2$, $M_1$, SPD-WL, GD-WL, run:

```bash
python test.py --mode s3
```

```bash
python test.py --mode s4
```

```bash
python test.py --mode n1
```

```bash
python test.py --mode n2
```

```bash
python test.py --mode m1
```

```bash
python test.py --mode distance
```

```bash
python test.py --mode resistance
```

You can also select only part of the graphs by specify:

```bash
python test.py --graph_type ($graph_type_selected)
```

The Category-to-range dict is:

```python
  "Basic": (0, 60),
  "Regular": (60, 160),
  "Extension": (160, 260),
  "CFI": (260, 360),
  "4-Vertex_Condition": (360, 380),
  "Distance_Regular": (380, 400),
  "Reliability": (400, 800)
```

Noting that only correct num of Reliability is 0 (all pairs are essentially the same) is reliable result.
