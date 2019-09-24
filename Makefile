.PHONY: serve

%.html: %.md template-revealjs.html
	pandoc -f markdown+fenced_code_blocks+smart \
		-t html5 \
		--template=template-revealjs.html \
		--standalone --section-divs \
		--variable theme="black" \
		--variable transition="none" \
		$< -o $@

serve:
	python3 -m http.server&

remark:
	git clone https://github.com/gnab/remark/
