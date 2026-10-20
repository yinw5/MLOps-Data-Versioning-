## Instructions on running

```bash
# fist time setup via CLI: Docker without Docker Desktop
brew install colima docker docker-compose
colima start --cpu 4 --memory 6 --vm-type vz --vz-rosetta
#  (in ~/.docker/config.json: "cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"])

# Marquez (cloned to ~/Downloads/MLops/tools/marquez, tag 0.51.1).
cd ~/Downloads/MLops/tools/marquez
./docker/up.sh --tag 0.51.1 --api-port 9000 --api-admin-port 9001 --detach --no-search

# data + pipeline
mkdir -p data/raw/airbnb && cp ~/Downloads/listings.csv ~/Downloads/reviews.csv data/raw/airbnb/
uv run python partb/pipeline.py
# lineage UI: http://localhost:3000 -> Jobs -> build_report

# when done:  ./docker/down.sh  (in the marquez folder)  and  colima stop
```