# Build paper PDF (requires a LaTeX distribution with pdflatex)
.PHONY: pdf clean

pdf: main.pdf

main.pdf: main.tex
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
