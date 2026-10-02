# Intent router with self-hosted Laya System 1.
#
# Build args:
#   PRELOAD_MODEL=true   download the Laya checkpoint into the image so
#                        cold starts don't fetch weights at runtime
#                        (adds ~1.6GB for laya-typed-decisions).
FROM python:3.12-slim

WORKDIR /srv/app
COPY requirements.txt .

# CPU torch first: the default CUDA wheel is ~2.5GB and unnecessary for
# a 421M decision model. Then the app deps (incl. the laya SDK).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir laya

COPY data/ ./data/
COPY src/ ./src/

ARG PRELOAD_MODEL=false
ARG LAYA_CHECKPOINT=convaiinnovations/laya-typed-decisions
ENV LAYA_CHECKPOINT=${LAYA_CHECKPOINT} \
    LAYA_PRELOAD=true \
    PORT=8080 \
    SYSTEM1_BACKEND=mock \
    SYSTEM2_BACKEND=mock \
    LOG_LEVEL=INFO \
    HF_HOME=/srv/app/.hf-cache
RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
      python -c "import laya; laya.load('$LAYA_CHECKPOINT')"; \
    fi

EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
