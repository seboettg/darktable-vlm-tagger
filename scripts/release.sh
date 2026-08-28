#!/usr/bin/env bash
#
# Cut a release for darktable-vlm-tagger.
#
#   scripts/release.sh 1.1.0
#
# Steps, in order:
#   1. bump `version` in pyproject.toml on `develop` and push
#   2. merge `develop` -> `main` (--no-ff), annotated tag `vX.Y.Z`, push
#   3. create the GitHub release from `main` with auto-generated notes
#   4. fast-forward `develop` back to `main` and push
#
# Requirements: a clean working tree, a local `develop` branch, and `gh`
# already authenticated (`gh auth status`).

set -euo pipefail

version="${1:?usage: scripts/release.sh X.Y.Z}"
tag="v${version}"
cd "$(git rev-parse --show-toplevel)"

# Pull only when the current branch actually tracks a remote (a fresh local
# `develop` won't until its first push).
pull_if_tracked() {
    if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
        git pull --ff-only
    fi
}

# --- preflight --------------------------------------------------------------
command -v gh >/dev/null || { echo "gh (GitHub CLI) not found" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || {
    echo "gh is not authenticated - run: gh auth login" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || {
    echo "working tree is not clean - commit or stash first" >&2; exit 1; }
git rev-parse --verify --quiet develop >/dev/null || {
    echo "no local 'develop' branch - create it with: git branch develop main" >&2
    exit 1; }
if git rev-parse --verify --quiet "refs/tags/${tag}" >/dev/null; then
    echo "tag ${tag} already exists" >&2; exit 1
fi

if command -v python3 >/dev/null && python3 -c 'import pytest' 2>/dev/null; then
    echo ">> running tests"
    python3 -m pytest -q
else
    echo ">> pytest not available, skipping tests" >&2
fi

# --- 1. bump version on develop -------------------------------------------
git switch develop
pull_if_tracked

perl -pi -e "s/^version = .*/version = \"${version}\"/" pyproject.toml
if git diff --quiet -- pyproject.toml; then
    echo ">> pyproject.toml already at ${version} (or has no 'version =' line)"
else
    git commit -m "Release ${version}" -- pyproject.toml
fi
git push origin develop

# --- 2. merge to main, tag, push ----------------------------------------
git switch main
pull_if_tracked
git merge --no-ff develop -m "Release ${version}"
git tag -a "${tag}" -m "Release ${version}"
git push origin main "${tag}"

# --- 3. GitHub release --------------------------------------------------
gh release create "${tag}" --target main --title "${version}" --generate-notes

# --- 4. keep develop in sync ------------------------------------------
git switch develop
git merge --ff-only main
git push origin develop

echo
echo "released ${tag}: $(gh release view "${tag}" --json url -q .url)"
