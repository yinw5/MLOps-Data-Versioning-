# Data Versioning Assignment — DVC + Git LFS

```
data-versioning-assignment/
├── pyproject.toml / uv.lock   # Python env (uv): pandas, sklearn, TF 2.15 + TF Privacy 0.9, dvc, ruff, pylint
├── data/raw/athletes.csv      # original download (never modified)
├── src/                       # master copy of the code (copied into each tool repo)
│   ├── common.py              # target, excluded cols, split ratio, preprocessing, metrics
│   ├── prepare_data.py        # builds v1 or v2 -> data/train.csv + data/test.csv   (tasks 1-3)
│   ├── eda.py                 # EDA of whichever version is checked out            (tasks 5, 9)
│   ├── train.py               # baseline LinearRegression, NEVER edited between v1/v2 (6-7, 10-11)
│   ├── dp_train.py            # non-DP vs DP-SGD NN on v2 + epsilon                (tasks 13-15)
│   └── compare.py             # prints the comparison tables                       (tasks 8/12, 15)
├── dvc-project/               # (you create) git repo versioned with DVC
├── dvc-storage/               # (you create) DVC "remote" (a local folder)
└── gitlfs-project/            # (you create) git repo versioned with Git LFS
```

Design decisions
- **v1** = the raw dataset, only adding `total_lift` (rows where it can't be computed are dropped — no target).
- **v2** = the assignment's cleaning code (only fix: `v2['deadlift']` → `data['deadlift']`, which would
  otherwise raise NameError) + new features: `bmi`, `howlong_years`, 8 survey flags.
- `total_lift = deadlift + candj + snatch + backsq`; those 4 columns are excluded from features (leakage).
- Same 80/20 split, `random_state=42` for both versions.
- `train.py` discovers columns from the data, so it runs unchanged on v1 and v2.

All commands below are run from inside the tool's repo folder.

---

## Tool 1 — DVC

```bash
cd ~/Downloads/MLops/data-versioning-assignment
mkdir dvc-project && cp -r src dvc-project/ && cd dvc-project

# Install: DVC is already in the uv env (`uv run dvc --version`). Alternative: `brew install dvc`.
git init
uv run dvc init
uv run dvc remote add -d localstore ../dvc-storage
printf "__pycache__/\n" >> .gitignore
git add -A && git commit -m "init DVC project + code"

# Tasks 1-4: create + version v1
uv run python src/prepare_data.py --version v1 --raw ../data/raw/athletes.csv
uv run dvc add data
git add data.dvc .gitignore && git commit -m "dataset v1 (raw)" && git tag v1
uv run dvc push

# Tasks 2-4: create + version v2
uv run python src/prepare_data.py --version v2 --raw ../data/raw/athletes.csv
uv run dvc add data
git add data.dvc && git commit -m "dataset v2 (cleaned + features)" && git tag v2
uv run dvc push
git log --oneline            # screenshot: two dataset versions

# Tasks 5-7: switch to v1, EDA + baseline
git checkout v1 -- data.dvc && uv run dvc checkout
uv run python src/eda.py --name v1        # -> reports/eda_v1/
uv run python src/train.py --name v1      # -> metrics/v1.json

# Tasks 8-11: switch to v2 (code untouched), EDA + model
git checkout v2 -- data.dvc && uv run dvc checkout
git status                                 # shows only data.dvc changed — proof code is unchanged
uv run python src/eda.py --name v2
uv run python src/train.py --name v2

# Task 12: compare
uv run python src/compare.py

# Task 13: linter
uv run ruff check src
uv run pylint src

# Tasks 14-15: DP on v2
uv run python src/dp_train.py --name v2 --noise 0.5,1.1,2.0   # ~1 min
uv run python src/compare.py

git add metrics reports && git commit -m "results"
```

## Tool 2 — Git LFS

```bash
brew install git-lfs           # install (note how easy/hard for the comparison slide)
git lfs install                # one-time: registers LFS filters in ~/.gitconfig

cd ~/Downloads/MLops/data-versioning-assignment
mkdir gitlfs-project && cp -r src gitlfs-project/ && cd gitlfs-project
git init
git lfs track "data/*.csv"     # creates .gitattributes
printf "__pycache__/\n" >> .gitignore
git add -A && git commit -m "init Git LFS project + code"

# v1
uv run python src/prepare_data.py --version v1 --raw ../data/raw/athletes.csv
git add data && git commit -m "dataset v1 (raw)" && git tag v1
git lfs ls-files               # screenshot: CSVs stored as LFS objects

# v2
uv run python src/prepare_data.py --version v2 --raw ../data/raw/athletes.csv
git add data && git commit -m "dataset v2 (cleaned + features)" && git tag v2
git log --oneline

# switch to v1 -> EDA + train
git checkout v1 -- data
uv run python src/eda.py --name v1
uv run python src/train.py --name v1

# switch to v2 -> EDA + train
git checkout v2 -- data
uv run python src/eda.py --name v2
uv run python src/train.py --name v2
uv run python src/compare.py

uv run ruff check src && uv run pylint src
uv run python src/dp_train.py --name v2 --noise 0.5,1.1,2.0
uv run python src/compare.py
git add metrics reports && git commit -m "results"
```

(Optional) pushing to GitHub: create an empty repo, then `git remote add origin <url>` and
`git push -u origin main --tags` — for Git LFS, the CSVs are uploaded to GitHub's LFS storage
(free quota 1 GB storage/bandwidth per month — fine here). DVC would need a separate remote (S3/GDrive/etc.).
