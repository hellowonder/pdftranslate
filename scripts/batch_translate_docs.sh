#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/batch_translate_docs.sh <input-dir-or-list-file> [output-root] [extra translator args...]
  scripts/batch_translate_docs.sh <input-dir-or-list-file> [--] [extra translator args...]

Description:
  If the input is a directory, recursively translate all .pdf, .djvu, .djv,
  and .epub files under it.

  If the input is a file, read it as a document list. Blank lines and lines
  starting with "#" or "//" are ignored. Other lines are document paths to
  translate. Relative paths are resolved from the list file's directory.

Defaults passed to PDF/DjVu translation:
  --generate-interleave-pdf
  --translation-model translategemma-4b-it
  --translation-profile translategemma
  --translation-base-url http://localhost:8001/v1
  --ocr-base-url http://localhost:8000/v1
  --ocr-model chandra
  --translation-latex-formula-handling direct

Defaults passed to EPUB translation:
  --translation-model translategemma-4b-it
  --translation-profile translategemma
  --translation-base-url http://localhost:8001/v1

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

shared_translation_args=(
  --translation-workers 32
  --translation-base-url http://localhost:8001/v1
  --translation-model translategemma-4b-it
  --translation-profile translategemma
)

pdf_default_args=(
  --generate-interleave-pdf
  --ocr-base-url http://localhost:8000/v1
  --ocr-model chandra
  --translation-latex-formula-handling direct
  --annotation-mode none
  --translation-scope page
)

epub_default_args=()

safe_output_name() {
  perl -CS -Mutf8 -pe '
    s/[^\p{L}\p{N}_-]+/_/g;
    s/_+/_/g;
    s/^_+|_+$//g;
  ' <<< "$1"
}

input_files=()

is_translated_input() {
  local path_name
  path_name=$(basename "$1")
  path_name=${path_name,,}

  [[ "$path_name" == *_interleaved.pdf ]] \
    || [[ "$path_name" == *_interleaved.epub ]] \
    || [[ "$path_name" == *_cn.pdf ]] \
    || [[ "$path_name" == *_cn.epub ]] \
    || [[ "$path_name" == *.cropped.pdf ]]
}

pdf_outputs_exist() {
  local output_dir=$1
  compgen -G "$output_dir/*_interleaved.pdf" > /dev/null
}

epub_outputs_exist() {
  local interleaved_output=$1
  local cn_output=$2
  [[ -f "$interleaved_output" || -f "$cn_output" ]]
}

if [[ "$input_mode" == "directory" ]]; then
  mapfile -d '' input_files < <(
    find "$input_root" \
      \( -type d \( -iname 'translated_output' -o -iname 'translate' -o -iname 'render' -o -iname 'ocr' \) -prune \) \
      -o \
      \( -type f \( -iname '*.pdf' -o -iname '*.djvu' -o -iname '*.djv' -o -iname '*.epub' \) -print0 \) | sort -z
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
      resolved_path=$(realpath -m "$line")
    else
      resolved_path=$(realpath -m "$input_root/$line")
    fi

    if is_translated_input "$resolved_path"; then
      continue
    fi

    input_files+=("$resolved_path")
  done < "$list_file"
fi

if [[ "$input_mode" == "directory" ]]; then
  filtered_files=()
  for input_path in "${input_files[@]}"; do
    if is_translated_input "$input_path"; then
      continue
    fi
    filtered_files+=("$input_path")
  done
  input_files=("${filtered_files[@]}")
fi

if [[ ${#input_files[@]} -eq 0 ]]; then
  if [[ "$input_mode" == "directory" ]]; then
    echo "No PDF, DjVu, or EPUB files found under: $input_root" >&2
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

  input_ext=${input_path##*.}
  input_ext=${input_ext,,}

  if [[ "$input_ext" == "epub" ]]; then
    interleaved_output="$file_output_dir/$(basename "${rel_no_ext}")_interleaved.epub"
    cn_output="$file_output_dir/$(basename "${rel_no_ext}")_cn.epub"

    echo "    Translator:  epub_translate.py"
    echo "    Output file: $interleaved_output"

    if epub_outputs_exist "$interleaved_output" "$cn_output"; then
      echo "    Skipped: existing EPUB output detected"
      success_count=$((success_count + 1))
      continue
    fi

    if ./.venv/bin/python src/translate/epub_translate.py \
      --input "$input_path" \
      --output "$interleaved_output" \
      --output-cn "$cn_output" \
      "${shared_translation_args[@]}" \
      "${epub_default_args[@]}" \
      "${extra_args[@]}"; then
      success_count=$((success_count + 1))
    else
      failure_count=$((failure_count + 1))
      echo "Translation failed: $input_path" >&2
    fi
    continue
  fi

  echo "    Translator:  pdf_translate.py"

  if pdf_outputs_exist "$file_output_dir"; then
    echo "    Skipped: existing PDF output detected"
    success_count=$((success_count + 1))
    continue
  fi

  if ./.venv/bin/python src/translate/pdf_translate.py \
    --input "$input_path" \
    --output-dir "$file_output_dir" \
    "${shared_translation_args[@]}" \
    "${pdf_default_args[@]}" \
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
