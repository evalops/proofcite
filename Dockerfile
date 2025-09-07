# minimal, reproducible runner for ProofCite
FROM python:3.11-slim

WORKDIR /app

# System deps (none heavy for this demo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/*

# This Dockerfile lives inside the proofcite/ directory.
# Copy the package contents into /app/proofcite
COPY . /app/proofcite
RUN cp /app/proofcite/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PROOFCITE_DOCS="proofcite/examples/regulatory/*.txt"

EXPOSE 7860
CMD ["python", "-m", "proofcite.gradio_app"]
