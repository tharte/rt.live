VERSION := $(shell git tag | tail -1)
COMMIT  := $(shell git log | grep commit | head -1 | cut -f2 -d' ') 

SED_SCRIPT  := 's/\\date{\\today}/\\date{\\today\\hspace{0.5cm}Build\: {\\tt version_from_shell (commit_from_shell\\\!\\\!\\\!)}}/g'
SED_SCRIPT  := $(subst version_from_shell,$(VERSION),$(SED_SCRIPT))
SED_SCRIPT  := $(subst commit_from_shell,$(COMMIT),$(SED_SCRIPT))

rt-live:     rt-live.org 
	cp -p $@.tex tmp.tex
	sed --in-place $(SED_SCRIPT) tmp.tex
	# render browser Babel source blocks as HTML
	sed --in-place 's/\]{browser}/\]{html}/g' tmp.tex
	pdflatex -shell-escape tmp.tex
	bibtex tmp
	pdflatex -shell-escape tmp.tex
	mv tmp.pdf $@.pdf
	# rm -f tmp.tex

clean:
	rm -f rt-live.aux
	rm -f rt-live.bbl
	rm -f rt-live.blg
	rm -f rt-live.log
	rm -f rt-live.toc
	rm -f rt-live.out

scratch:     scratch.tex
	pdflatex -shell-escape $<
