#!/usr/bin/env bash
set -euo pipefail

resolve_source_revision() {
  local event_revision=$1
  local repository=$2
  local target=$3
  local release_tag=$4
  local tag_ref tag_type tag_object tag_target target_type source_revision

  if [[ ! "$release_tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Release tag $release_tag is not a stable semantic version" >&2
    return 1
  fi

  if [[ "$target" == "testpypi" ]]; then
    git rev-parse --verify "$event_revision^{commit}"
    return
  fi
  if [[ "$target" != "pypi" ]]; then
    echo "Unknown release target $target" >&2
    return 1
  fi

  tag_ref="$(
    gh api "repos/$repository/git/ref/tags/$release_tag" \
      --jq '[.object.type, .object.sha] | @tsv'
  )"
  IFS=$'\t' read -r tag_type tag_object <<< "$tag_ref"
  if [[ "$tag_type" != "tag" || -z "$tag_object" ]]; then
    echo "Production release tag $release_tag must be annotated" >&2
    return 1
  fi

  tag_target="$(
    gh api "repos/$repository/git/tags/$tag_object" \
      --jq '[.object.type, .object.sha] | @tsv'
  )"
  IFS=$'\t' read -r target_type source_revision <<< "$tag_target"
  if [[ "$target_type" != "commit" || -z "$source_revision" ]]; then
    echo "Release tag $release_tag does not point directly to a commit" >&2
    return 1
  fi
  printf '%s\n' "$source_revision"
}

qualify_release() {
  local event_revision=$1
  local repository=$2
  local target=$3
  local release_tag=$4
  local source_revision
  local ci_run
  local run_id head_sha head_branch event status conclusion
  local ci_success_count

  source_revision=$(resolve_source_revision \
    "$event_revision" "$repository" "$target" "$release_tag")
  source_revision=$(git rev-parse --verify "$source_revision^{commit}")
  if [[ ! "$source_revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Release source did not resolve to a full commit SHA" >&2
    return 1
  fi
  if ! git merge-base --is-ancestor \
    "$source_revision" refs/remotes/origin/main; then
    echo "Release commit $source_revision is not on origin/main" >&2
    return 1
  fi

  ci_run="$(
    gh api \
      "repos/$repository/actions/workflows/ci.yml/runs?branch=main&event=push&head_sha=$source_revision&status=success&per_page=100" \
      --jq '(.workflow_runs[0] // empty) | [.id, .head_sha, .head_branch, .event, .status, .conclusion] | @tsv'
  )"
  if [[ -z "$ci_run" ]]; then
    echo "No successful main CI run exists for $source_revision" >&2
    return 1
  fi

  IFS=$'\t' read -r run_id head_sha head_branch event status conclusion \
    <<< "$ci_run"
  if [[ "$head_sha" != "$source_revision" \
    || "$head_branch" != "main" \
    || "$event" != "push" \
    || "$status" != "completed" \
    || "$conclusion" != "success" ]]; then
    echo "GitHub returned an ineligible CI run for $source_revision" >&2
    return 1
  fi

  ci_success_count="$(
    gh api \
      "repos/$repository/actions/runs/$run_id/jobs?filter=latest&per_page=100" \
      --jq '[.jobs[] | select(.name == "CI Success" and .status == "completed" and .conclusion == "success")] | length'
  )"
  if [[ "$ci_success_count" -lt 1 ]]; then
    echo "CI run $run_id has no successful CI Success job" >&2
    return 1
  fi
  printf '%s\n' "$source_revision"
}

run_fixture() (
  local target=$1
  local release_tag=$2
  local on_main=$3
  local run=$4
  local job_count=$5
  local fixture_tag_ref=$6
  local fixture_tag_target=$7
  local fixture_sha=1111111111111111111111111111111111111111

  git() {
    if [[ "$1" == "rev-parse" ]]; then
      local revision=${3%%^*}
      if [[ "$revision" == "fixture-ref" ]]; then
        revision=$fixture_sha
      fi
      printf '%s\n' "$revision"
    else
      [[ "$on_main" == "yes" ]]
    fi
  }
  gh() {
    case "$2" in
      */git/ref/tags/*) printf '%s\n' "$fixture_tag_ref" ;;
      */git/tags/*) printf '%s\n' "$fixture_tag_target" ;;
      */workflows/ci.yml/runs?*) printf '%s\n' "$run" ;;
      */jobs?*) printf '%s\n' "$job_count" ;;
      *) return 1 ;;
    esac
  }
  qualify_release fixture-ref example/advect "$target" "$release_tag"
)

self_test() {
  local fixture_sha=1111111111111111111111111111111111111111
  local eligible_run="42"$'\t'"$fixture_sha"$'\tmain\tpush\tcompleted\tsuccess'
  local annotated_tag=$'tag\tfixture-tag-object'
  local commit_target="commit"$'\t'"$fixture_sha"

  run_fixture \
    testpypi v1.2.3 yes "$eligible_run" 1 "$annotated_tag" "$commit_target" \
    >/dev/null
  run_fixture \
    pypi v1.2.3 yes "$eligible_run" 1 "$annotated_tag" "$commit_target" \
    >/dev/null

  if run_fixture \
    pypi 'v1.2.3;echo unsafe' yes "$eligible_run" 1 \
    "$annotated_tag" "$commit_target" >/dev/null 2>&1; then
    echo "An invalid tag name passed the production release guard" >&2
    return 1
  fi

  if run_fixture \
    pypi v1.2.3 yes "$eligible_run" 1 \
    "commit"$'\t'"$fixture_sha" "$commit_target" >/dev/null 2>&1; then
    echo "A lightweight tag passed the production release guard" >&2
    return 1
  fi

  if run_fixture \
    pypi v1.2.3 yes "$eligible_run" 1 \
    "$annotated_tag" $'tag\tnested-tag-object' >/dev/null 2>&1; then
    echo "An annotated tag not pointing to a commit passed the release guard" >&2
    return 1
  fi

  if run_fixture \
    pypi v1.2.3 no "$eligible_run" 1 "$annotated_tag" "$commit_target" \
    >/dev/null 2>&1; then
    echo "An off-main revision passed the release guard" >&2
    return 1
  fi

  if run_fixture \
    pypi v1.2.3 yes \
    $'42\t2222222222222222222222222222222222222222\tmain\tpush\tcompleted\tsuccess' \
    1 "$annotated_tag" "$commit_target" >/dev/null 2>&1; then
    echo "CI for another revision passed the release guard" >&2
    return 1
  fi

  if run_fixture \
    pypi v1.2.3 yes "$eligible_run" 0 "$annotated_tag" "$commit_target" \
    >/dev/null 2>&1; then
    echo "A run without CI Success passed the release guard" >&2
    return 1
  fi
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
elif [[ $# -eq 4 ]]; then
  qualify_release "$1" "$2" "$3" "$4"
else
  echo "usage: $0 REVISION OWNER/REPOSITORY TARGET TAG | --self-test" >&2
  exit 2
fi
