.PHONY: install dev test demo cli dspy gradio eval docker-build docker-run

install:
	python3 -m pip install -U pip
	python3 -m pip install -e .[dspy]

demo:
	python3 proofcite/examples/comprehensive_demo.py

cli:
	python3 -m proofcite.cli --docs "proofcite/examples/data/*.txt" --q "What port does Jellyfin use?" --json

dspy:
	python3 -m proofcite.dspy_cli --docs "proofcite/examples/data/*.txt" --q "What port does Jellyfin use?" --json

gradio:
	python3 -m proofcite.gradio_app

eval:
	python3 -m proofcite.examples.evaluate_devset --mode baseline --docs "proofcite/examples/data/*.txt"

docker-build:
	docker build -t evalops/proofcite .

docker-run:
	docker run --rm -p 7860:7860 evalops/proofcite

