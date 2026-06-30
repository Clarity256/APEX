# WCNC 2027 Chinese Draft

This directory is the Chinese writing companion for the English IEEE WCNC 2027
paper in `../wcnc2027`.

## Scope

The Chinese draft follows the same technical scope as the English version:

1. fast-timescale PRB--power fair allocation for fixed association;
2. slow-timescale satellite-cell association with hard per-cell handover
   budgets;
3. exact and rolling-horizon optimization methods.

Online learning, behavior cloning, PPO, and hierarchical reinforcement learning
are future-work material for this conference version.

## Build

The default local build uses XeLaTeX plus BibTeX:

```bash
make
```

The bibliography database is shared with the English draft:

```text
../bib/leo_references.bib
```

Use `make clean` to remove generated files.
