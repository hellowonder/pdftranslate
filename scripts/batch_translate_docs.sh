#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/batch_translate_docs.sh <input-dir-or-list-file> [output-root] [extra pdf_translate args...]
  scripts/batch_translate_docs.sh <input-dir-or-list-file> [--] [extra pdf_translate args...]

Description:
  If the input is a directory, recursively translate all .pdf, .djvu, and .djv
  files under it.

  If the input is a file, read it as a document list. Blank lines and lines
  starting with "#" or "//" are ignored. Other lines are document paths to
  translate. Relative paths are resolved from the list file's directory.

Defaults passed to pdf_translate.py:
  --generate-interleave-pdf
  --translation-base-url http://localhost:11434/v1
  --translation-model gemma4:26b
  --ocr-base-url http://localhost:11434/v1
  --ocr-model deepseek-ocr:3b
  --translation-latex-formula-handling direct

Output layout:
  Each source file gets its own output directory under [output-root].
  If [output-root] is omitted, it defaults to:
    directory input: <input-dir>/translated_output
    list file input: <list-file-dir>/translated_output

Examples:
  scripts/batch_translate_docs.sh ./books
  scripts/batch_translate_docs.sh /mnt/d/mathbooks/textbook/toread.txt
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

input_arg=$1
shift

if [[ ! -e "$input_arg" ]]; then
  echo "Input does not exist: $input_arg" >&2
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

input_mode=""
input_root=""
list_file=""

if [[ -d "$input_arg" ]]; then
  input_mode="directory"
  input_root=$(realpath "$input_arg")
elif [[ -f "$input_arg" ]]; then
  input_mode="list"
  list_file=$(realpath "$input_arg")
  input_root=$(dirname "$list_file")
else
  echo "Input must be a directory or a document list file: $input_arg" >&2
  exit 1
fi

if [[ -z "$output_root" ]]; then
  output_root="$input_root/translated_output"
fi
output_root=$(realpath -m "$output_root")

mkdir -p "$output_root"

default_args=(
  --translation-workers 32
  --generate-interleave-pdf
  --translation-base-url http://192.168.3.105:11434/v1
  --translation-model gemma4:26b
  --ocr-base-url http://localhost:8000/v1
  --ocr-model chandra
  --translation-latex-formula-handling direct
  --annotation-mode page
  --translation-scope page
)

safe_output_name() {
  perl -CS -Mutf8 -pe '
    s/[^\p{L}\p{N}_-]+/_/g;
    s/_+/_/g;
    s/^_+|_+$//g;
  ' <<< "$1"
}

input_files=()

if [[ "$input_mode" == "directory" ]]; then
  mapfile -d '' input_files < <(
    find "$input_root" -type f \( -iname '*.pdf' -o -iname '*.djvu' -o -iname '*.djv' \) -print0 | sort -z
  )
else
  while IFS= read -r line || [[ -n "$line" ]]; do
    line=${line//$'\r'/}
    line=${line#"${line%%[![:space:]]*}"}
    line=${line%"${line##*[![:space:]]}"}

    if [[ -z "$line" || "$line" == \#* || "$line" == //* ]]; then
      continue
    fi

    if [[ "$line" = /* ]]; then
      input_files+=("$(realpath -m "$line")")
    else
      input_files+=("$(realpath -m "$input_root/$line")")
    fi
  done < "$list_file"
fi

if [[ ${#input_files[@]} -eq 0 ]]; then
  if [[ "$input_mode" == "directory" ]]; then
    echo "No PDF or DjVu files found under: $input_root" >&2
  else
    echo "No input files found in list: $list_file" >&2
  fi
  exit 0
fi

if [[ "$input_mode" == "directory" ]]; then
  echo "Found ${#input_files[@]} document(s) under $input_root"
else
  echo "Found ${#input_files[@]} document(s) in $list_file"
fi
echo "Output root: $output_root"

success_count=0
failure_count=0

for input_path in "${input_files[@]}"; do
  if [[ ! -f "$input_path" ]]; then
    failure_count=$((failure_count + 1))
    echo "Input file does not exist: $input_path" >&2
    continue
  fi

  rel_path=${input_path#"$input_root"/}
  rel_no_ext=${rel_path%.*}
  rel_no_ext=$(safe_output_name "$rel_no_ext")
  if [[ -z "$rel_no_ext" ]]; then
    rel_no_ext="document"
  fi
  file_output_dir="$output_root/$rel_no_ext"
  mkdir -p "$file_output_dir"

  echo
  echo "==> Translating: $input_path"
  echo "    input_root: $input_root"
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
