#!/bin/bash
# Re-sync the engine from 06_recommend, upload the app to the team workspace, and redeploy.
# Usage: tools/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."
P=datathon
DEST="/Workspace/Shared/Datathon 2026/app"
.venv/bin/python tools/sync_engine.py
.venv/bin/python -c "import ast; ast.parse(open('engine/recommend_engine.py').read())"
STAGE=$(mktemp -d); mkdir -p "$STAGE/engine" "$STAGE/tools"
cp app.py data.py services.py capabilities.py ui_results.py versions.py app.yaml requirements.txt "$STAGE/"; mkdir -p "$STAGE/catalog" "$STAGE/config" "$STAGE/.streamlit"; cp catalog/capabilities.json catalog/conditions.json catalog/model1_levels.csv catalog/model1_services.json catalog/model2_capability.parquet "$STAGE/catalog/"; cp config/*.json "$STAGE/config/"; cp .streamlit/config.toml "$STAGE/.streamlit/"
cp engine/*.py "$STAGE/engine/"; cp tools/sync_engine.py tools/deploy.sh "$STAGE/tools/"
databricks workspace import-dir "$STAGE" "$DEST" --overwrite --profile $P > /dev/null
databricks apps deploy carecompass --source-code-path "$DEST" --profile $P -o json | python3 -c "import sys,json; print('deploy:', json.load(sys.stdin)['status']['state'])"
rm -rf "$STAGE"
