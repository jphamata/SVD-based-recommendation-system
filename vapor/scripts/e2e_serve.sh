#!/bin/sh
# Serve a model (checkpoint directory or .gguf), talk to it over the OpenAI
# API with curl (plain and streaming, chat and completion), check the
# answers, stop the server.
set -eu
model=$1
port=${PORT:-8765}
log=$(mktemp)
out=$(mktemp)
if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
  echo "port $port is already serving; set PORT" >&2; exit 1
fi
mix vapor.serve --model "$model" --port "$port" > "$log" 2>&1 &
pid=$!
# the server is gone (port free) before the next run starts
trap 'kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true; rm -f "$log" "$out"' EXIT
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && break
  kill -0 $pid 2>/dev/null || { cat "$log" >&2; exit 1; }
  sleep 0.5
done
base="http://127.0.0.1:$port/v1"
echo "--- GET /v1/models"; curl -sf "$base/models"; echo
echo "--- POST /v1/completions"
curl -sf "$base/completions" -H 'content-type: application/json' \
  -d '{"prompt": "Hello", "max_tokens": 16, "temperature": 0}' > "$out"
cat "$out"; echo
grep -q '"text_completion"' "$out"
echo "--- POST /v1/chat/completions (stream)"
curl -sfN "$base/chat/completions" -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Oi!"}], "max_tokens": 16, "temperature": 0.7, "seed": 2, "stream": true}' > "$out"
cat "$out"
grep -q 'data: \[DONE\]' "$out"
grep -q '"finish_reason":"length"' "$out"
echo "e2e: ok ($model)"
