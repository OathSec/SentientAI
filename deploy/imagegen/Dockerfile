# Fuzzy's image server: Stable Diffusion (diffusers, full precision) behind an OpenAI-style API.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
RUN pip install --no-cache-dir \
      diffusers==0.30.3 transformers==4.44.2 accelerate==0.34.2 huggingface_hub==0.25.2 \
      safetensors==0.4.5 fastapi==0.115.0 uvicorn==0.30.6
WORKDIR /app
COPY server.py worker.py /app/
ENV HF_HOME=/models/hf PYTHONUNBUFFERED=1
EXPOSE 7861
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7861"]
