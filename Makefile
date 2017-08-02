.PHONY: serve

%.html: %.md template-revealjs.html
	pandoc -f markdown+fenced_code_blocks \
		-t html5 \
		--template=template-revealjs.html \
		--smart --standalone --section-divs \
		--variable theme="black" \
		--variable transition="none" \
		$< -o $@

serve:
	python3 -m http.server
