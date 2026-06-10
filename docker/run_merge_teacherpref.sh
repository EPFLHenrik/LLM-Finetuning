for dir in data/preprocessing/*/; do
    student="$dir/student_corrected.jsonl"
    teacher="$dir/teacher_corrected.jsonl"
    if [ ! -f "$student" ] || [ ! -f "$teacher" ]; then
        echo "Skipping $dir: missing student_corrected.jsonl or teacher_corrected.jsonl"
        continue
    fi
    echo "Processing $dir..."
    python training/merge_reasoning_from_student_teacher.py "$student" "$teacher" --prefer-teacher --output "$dir/merged_teacher_pref.jsonl"
done