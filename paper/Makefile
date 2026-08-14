TEX := main.tex

.PHONY: all watch clean

all:
	tectonic -X compile $(TEX)

watch:
	tectonic -X watch

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out
