#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/benchmarks
out="data/benchmarks/benchmark_prompts.jsonl"
: > "$out"

for i in $(seq 1 15); do
  a=$((i + 3))
  b=$((i * 2))
  ans=$((a + b))
  printf '{"id":"math-%03d","category":"math","prompt":"Calculate %d + %d","expected_keywords":["%d"],"required_tools":["calculator"],"current_events":false}\n' "$i" "$a" "$b" "$ans" >> "$out"
done

for i in $(seq 1 15); do
  printf '{"id":"reasoning-%03d","category":"reasoning","prompt":"If all roses are flowers and some flowers fade quickly, what can be concluded about roses? Case %d.","expected_keywords":["roses","flowers"],"required_tools":[],"current_events":false}\n' "$i" "$i" >> "$out"
done

for i in $(seq 1 15); do
  printf '{"id":"knowledge-%03d","category":"knowledge","prompt":"Who wrote The Odyssey? Variant %d.","expected_keywords":["Homer"],"required_tools":["wikipedia"],"current_events":false}\n' "$i" "$i" >> "$out"
done

for i in $(seq 1 15); do
  printf '{"id":"current-events-%03d","category":"current_events","prompt":"Find latest major AI regulation update this week, sample %d, include source.","expected_keywords":["source"],"required_tools":["duckduckgo_search","webpage_reader"],"current_events":true}\n' "$i" "$i" >> "$out"
done

for i in $(seq 1 15); do
  printf '{"id":"tool-use-%03d","category":"tool_use","prompt":"Convert %d kilometers to miles and explain method.","expected_keywords":["miles"],"required_tools":["unit_converter"],"current_events":false}\n' "$i" "$((i+10))" >> "$out"
done

for i in $(seq 1 15); do
  printf '{"id":"multi-step-%03d","category":"multi_step","prompt":"Compute (12 + %d) then convert result from USD to INR.","expected_keywords":["INR"],"required_tools":["calculator","currency_converter"],"current_events":false}\n' "$i" "$i" >> "$out"
done

for i in $(seq 1 15); do
  printf '{"id":"code-generation-%03d","category":"code_generation","prompt":"Write Python function %d to return fibonacci sequence first n numbers.","expected_keywords":["def","fibonacci"],"required_tools":[],"current_events":false}\n' "$i" "$i" >> "$out"
done

for i in $(seq 1 15); do
  printf '{"id":"document-qa-%03d","category":"document_qa","prompt":"Search local docs for LangGraph mention and summarize finding case %d.","expected_keywords":["LangGraph"],"required_tools":["document_search","local_rag"],"current_events":false}\n' "$i" "$i" >> "$out"
done

echo "Wrote $(wc -l < "$out") prompts to $out"
