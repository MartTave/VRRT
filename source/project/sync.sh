rsync -rzLv \
    --include='data/right_merged.mp4' \
    --include='data/right_merged_full.csv' \
    --exclude='data' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='results' \
    --exclude='debug' \
    --exclude='models/fine_tuned/new_dataset/' \
    ./ disco:~/datasets/project
