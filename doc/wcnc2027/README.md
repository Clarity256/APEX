# WCNC 2027 LaTeX Project

This directory is the IEEE conference LaTeX project for the LEO direct-access
downlink resource-allocation paper.

## Current Paper Scope

The WCNC 2027 draft is scoped to:

1. fast-timescale PRB--power fair allocation for fixed association;
2. slow-timescale satellite-cell association with hard per-cell handover
   budgets;
3. exact and rolling-horizon optimization methods.

Online scheduling, behavior cloning, PPO, and hierarchical reinforcement
learning are treated as future work rather than core contributions in this
6-page conference version.

Planned page budget:

- Introduction and compact related work: 0.8 pages.
- System model: 1.0 page.
- Problem formulation: 1.0 page.
- Proposed method: 1.2 pages.
- Simulation setup and results: 1.5 pages.
- Conclusion and references: 0.5 pages.

## Template Source

- IEEE Author Center conference tools page:
  <https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/>
- IEEE conference template download endpoint referenced by IEEE conference
  author pages:
  <https://www.ieee.org/content/dam/ieee-org/ieee/web/org/conferences/conference-latex-template.zip>
- Overleaf official IEEE conference template mirror:
  <https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn>

The official IEEE ZIP files have been archived under `../latex_templates/ieee/`:

- `../latex_templates/ieee/archives/conference-latex-template.zip`
- `../latex_templates/ieee/archives/IEEEtranBST2.zip`

For reproducible local builds, `IEEEtran.cls` and `IEEEtran.bst` are also copied
to this directory next to `main.tex`.

## Structure

- `main.tex`: paper entry point.
- `sections/`: section-level LaTeX files.
- `../bib/leo_references.bib`: shared BibTeX database for English and Chinese
  drafts.
- `figures/`: generated experiment figures.
- `IEEEtran.cls`: local IEEE conference class copied from the official ZIP.
- `IEEEtran.bst`: local IEEE bibliography style copied from the official ZIP.

## Build

The default local build uses PdfLaTeX plus BibTeX:

```bash
make
```

If a full TeX distribution with `latexmk` is installed, use:

```bash
make latexmk
```

Manual classic sequence:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Clean generated artifacts with:

```bash
make clean
```
