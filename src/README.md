## Instructions on running

Setup (macOS):
```bash
brew install uv git-lfs && git lfs install
uv sync                         # installs pandas, scikit-learn, TF 2.15, TF Privacy 0.9, dvc, ruff, pylint
mkdir -p data/raw && unzip athletes.csv.zip -d data/raw
```

Tool 1 – DVC (run inside `dvc/`):
```bash
uv run dvc init --subdir && uv run dvc remote add -d localstore ../dvc-storage
uv run python ../src/prepare_data.py --version v1 --raw ../data/raw/athletes.csv
uv run dvc add data && git add data.dvc .gitignore && git commit -m "v1" && git tag dvc-v1 && uv run dvc push
uv run python ../src/prepare_data.py --version v2 --raw ../data/raw/athletes.csv
uv run dvc add data && git add data.dvc && git commit -m "v2" && git tag dvc-v2 && uv run dvc push

git checkout dvc-v1 -- data.dvc && uv run dvc checkout      # switch to v1
uv run python ../src/eda.py --name v1 && uv run python ../src/train.py --name v1
git checkout dvc-v2 -- data.dvc && uv run dvc checkout      # switch to v2 (no code change)
uv run python ../src/eda.py --name v2 && uv run python ../src/train.py --name v2
uv run python ../src/compare.py
uv run python ../src/dp_train.py --name v2 --noise 0.5,1.1,2.0 && uv run python ../src/compare.py
```

Tool 2 – Git LFS (run inside `gitlfs/`; the tracking rule is in the repo-root `.gitattributes`):
```bash
git lfs track "gitlfs/data/*.csv"       # run from repo root
uv run python ../src/prepare_data.py --version v1 --raw ../data/raw/athletes.csv
git add data && git commit -m "v1" && git tag lfs-v1
uv run python ../src/prepare_data.py --version v2 --raw ../data/raw/athletes.csv
git add data && git commit -m "v2" && git tag lfs-v2

git checkout lfs-v1 -- data             # switch to v1
uv run python ../src/eda.py --name v1 && uv run python ../src/train.py --name v1
git checkout lfs-v2 -- data             # switch to v2 (no code change)
uv run python ../src/eda.py --name v2 && uv run python ../src/train.py --name v2
uv run python ../src/compare.py
uv run python ../src/dp_train.py --name v2 --noise 0.5,1.1,2.0 && uv run python ../src/compare.py
```