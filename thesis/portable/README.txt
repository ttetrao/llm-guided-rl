Portable source of thesis/doc.tex
================================

Self-contained: doc.tex plus every image it includes, under img/.
The paths in doc.tex are the originals from the repository with the
prefix ../src/output/ replaced by img/, so the layout under img/ mirrors
src/output/ and the mapping back is one prefix swap.

Build (needs a TeX Live with pdflatex + latexmk):

    latexmk -pdf doc.tex

Result: doc.pdf, 24 pages. No .bib file: the bibliography is inline in
doc.tex via thebibliography.
