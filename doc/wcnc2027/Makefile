PDFLATEX ?= pdflatex
BIBTEX ?= bibtex
LATEXMK ?= latexmk
MAIN := main
CLEAN_FILES := $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).fdb_latexmk \
	$(MAIN).fls $(MAIN).log $(MAIN).out $(MAIN).pdf $(MAIN).synctex.gz \
	$(MAIN).xdv

.PHONY: all latexmk clean

all:
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex

latexmk:
	$(LATEXMK) -pdf $(MAIN).tex

clean:
	rm -f $(CLEAN_FILES)
