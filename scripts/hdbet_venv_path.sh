# Shared location of the isolated HD-BET venv (see setup_hdbet_venv.sh for
# why it needs its own venv, separate from the main one).
#
# Both setup_hdbet_venv.sh and run_preprocess_brats.sh source this file
# instead of each hardcoding their own copy of the path -- edit it here ONCE
# (e.g. to move it onto larger storage) and every script stays in sync
# automatically. Can also be overridden per-shell via an exported
# HDBET_VENV_DIR env var without editing this file.
HDBET_VENV_DIR="${HDBET_VENV_DIR:-/media/storage/luu/hdbet_venv}"
