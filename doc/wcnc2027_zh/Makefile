XELATEX ?= xelatex
BIBTEX ?= bibtex
LATEXMK ?= latexmk
MAIN := main
CLEAN_FILES := $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).fdb_latexmk \
	$(MAIN).fls $(MAIN).log $(MAIN).out $(MAIN).pdf $(MAIN).synctex.gz \
	$(MAIN).toc $(MAIN).xdv

.PHONY: all latexmk clean

all:
	$(XELATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(XELATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(XELATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex

latexmk:
	$(LATEXMK) -xelatex $(MAIN).tex

clean:
	rm -f $(CLEAN_FILES)
