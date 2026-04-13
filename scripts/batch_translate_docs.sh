#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/batch_translate_docs.sh <input-dir> [output-root] [extra pdf_translate args...]
  scripts/batch_translate_docs.sh <input-dir> [--] [extra pdf_translate args...]

Description:
  Recursively translate all .pdf, .djvu, and .djv files under <input-dir>.

Defaults passed to pdf_translate.py:
  --generate-interleave-pdf
  --translation-base-url http://localhost:11434/v1
  --translation-model gemma4:26b
  --ocr-base-url http://localhost:11434/v1
  --ocr-model deepseek-ocr:3b
  --translation-latex-formula-handling direct

Output layout:
  Each source file gets its own output directory under [output-root].
  If [output-root] is omitted, it defaults to <input-dir>/translated_output.

Examples:
  scripts/batch_translate_docs.sh ./books
  scripts/batch_translate_docs.sh ./books ./outputs --translation-workers 32
  scripts/batch_translate_docs.sh ./books --translation-workers 32 --translation-model qwen2.5:72b
EOF
}

if [[ $# -gt 0 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

input_dir=$1
shift

if [[ ! -d "$input_dir" ]]; then
  echo "Input directory does not exist: $input_dir" >&2
  exit 1
fi

output_root=""
extra_args=()

if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then
    shift
    extra_args=("$@")
  elif [[ "$1" == -* ]]; then
    extra_args=("$@")
  else
    output_root=$1
    shift
    if [[ $# -gt 0 ]]; then
      if [[ "$1" == "--" ]]; then
        shift
      fi
      extra_args=("$@")
    fi
  fi
fi

input_dir=$(realpath "$input_dir")
if [[ -z "$output_root" ]]; then
  output_root="$input_dir/translated_output"
fi
output_root=$(realpath -m "$output_root")

mkdir -p "$output_root"

default_args=(
  --translation-workers 32
  --generate-interleave-pdf
  --translation-base-url http://localhost:11434/v1
  --translation-model gemma4:26b
  --ocr-base-url http://localhost:8000/v1
  --ocr-model deepseek-ocr:3b
  --translation-latex-formula-handling direct
  --enable-annotation
)

mapfile -d '' input_files < <(
  find "$input_dir" -type f \( -iname '*.pdf' -o -iname '*.djvu' -o -iname '*.djv' \) -print0 | sort -z
)

if [[ ${#input_files[@]} -eq 0 ]]; then
  echo "No PDF or DjVu files found under: $input_dir" >&2
  exit 0
fi

echo "Found ${#input_files[@]} document(s) under $input_dir"
echo "Output root: $output_root"

success_count=0
failure_count=0

for input_path in "${input_files[@]}"; do
  rel_path=${input_path#"$input_dir"/}
  rel_no_ext=${rel_path%.*}
  rel_no_ext=${rel_no_ext//[^a-zA-Z0-9_-]/_}
  file_output_dir="$output_root/$rel_no_ext"
  mkdir -p "$file_output_dir"

  echo
  echo "==> Translating: $input_path"
  echo "    input_dir: $input_dir"
  echo "    rel_path: $rel_path"
  echo "    rel_no_ext: $rel_no_ext"
  echo "    Output dir:  $file_output_dir"

  if ./.venv/bin/python src/translate/pdf_translate.py \
    --input "$input_path" \
    --output-dir "$file_output_dir" \
    "${default_args[@]}" \
    "${extra_args[@]}"; then
    success_count=$((success_count + 1))
  else
    failure_count=$((failure_count + 1))
    echo "Translation failed: $input_path" >&2
  fi
done

echo
echo "Completed. success=$success_count failure=$failure_count"

if [[ $failure_count -gt 0 ]]; then
  exit 1
fi
