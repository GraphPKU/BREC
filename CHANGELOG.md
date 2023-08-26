# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.0.0] - 2023-08-26

### Added

- Distribute Pypi package at [Pypi](https://pypi.org/project/brec).
- Add CHANGELOG.


## [0.3.0] - 2023-06-14

### Added

- Implement 6 GNN baseines: SparseWL, OSAN, SWL, GSN, DropGNN, Graphormer.
- Implement 2 non-GNN baselines: SPD-WL and GD-WL.
- Add MIT LICENSE.

### Changed

- [Breaking Changes] Update $T^2$-statistics threshold.
- Update guidelines.


## [0.2.0] - 2023-04-22

### Added

- Upload data files in Github.
- Add paper arxiv link at [arxiv](https://arxiv.org/abs/2304.07702)

### Changed

- Update guidelines.

### Fixed

- Correct data order for non-GNNs.


## [0.1.0] - 2023-04-16

### Added

- Release BREC dataset and evaluation framework.
- Implement 10 GNN baselines: NGNN, DE+NGNN, DS-GNN, DSS-GNN, SUN, GNN-AK, KP-GNN, I$^2$-GNN, PPGN and KC-SetGNN.
- Implement 6 non-GNN baselines: 3-WL, $S_3$, $S_4$, $N_1$, $N_2$ and $M_1$. (WL, $S_k$ and $N_k$ also extensible)