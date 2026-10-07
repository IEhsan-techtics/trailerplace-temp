#!/usr/bin/env bash
# Build the Luna image and create (or update) the hourly follow-up job.
#
#   ./deploy/followup-job.sh <image-tag>
#
# The job runs `python -m src.followup` once an hour (src/followup): it finds Messenger
# customers who went quiet after our message, lets the model decide and write a follow-up,
# sends it and records it in the conversation.
#
# It takes every setting from the live trailerplace-api app, so it reads the same database,
# model and Messenger page as the bot, then adds FOLLOWUP_ENABLED=1. The page token is copied
# as a job secret, never as a plain variable. trailerplace-api itself is not changed.
set -euo pipefail

TAG="${1:?usage: deploy/followup-job.sh <image-tag>}"
RG="TrailerPlace-Resource"
APP="trailerplace-api"
JOB="trailerplace-followup"
REGISTRY="TrailerPlaceBackend"
SERVER="trailerplacebackend-crhuewcedpbre8fz.azurecr.io"
ENVIRONMENT="managedEnvironment-TrailerPlaceRes-812b"
IMAGE="$SERVER/trailerplace-api:$TAG"
SCHEDULE="${FOLLOWUP_CRON:-0 * * * *}"   # every hour, on the hour (UTC)
PY="${PYTHON:-python}"

cd "$(dirname "$0")/.."
# The build log carries characters a Windows console code page cannot print, and az crashes
# on them mid-stream (the build itself carries on in Azure).
export PYTHONIOENCODING=utf-8
# Git Bash rewrites environment values that look like Unix paths when it starts a Windows
# program, so the environment id "/subscriptions/..." reached the spec as a disk path and
# Azure answered "Environment not found". The FJ_* values are left exactly as read.
export MSYS2_ENV_CONV_EXCL="FJ_"

if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  echo "== using the existing image $IMAGE"
else
  echo "== building $IMAGE in Azure"
  # Queued without streaming the log: az crashes printing it on a Windows console (the
  # build carries on regardless), so the run is polled to its end instead.
  RUN_ID="$(az acr build --registry "$REGISTRY" --image "trailerplace-api:$TAG" --file Dockerfile.backend . \
    --no-logs --only-show-errors --query runId -o tsv | tr -d '\r')"
  [[ -n "$RUN_ID" ]] || { echo "ERROR: the build was not queued" >&2; exit 1; }
  while :; do
    STATUS="$(az acr task show-run -r "$REGISTRY" --run-id "$RUN_ID" --query status -o tsv 2>/dev/null | tr -d '\r')"
    case "$STATUS" in
      Succeeded) echo "   build $RUN_ID succeeded"; break ;;
      Failed|Canceled|Error|Timeout) echo "ERROR: build $RUN_ID ended $STATUS" >&2; exit 1 ;;
      *) sleep 20 ;;
    esac
  done
fi

echo "== reading $APP's settings"
# Passed to the spec builder through the environment, never on a command line.
# tr: az on Windows ends each value with a carriage return Azure then rejects.
# The page token is whatever the API's MESSENGER_PAGE_ACCESS_TOKEN actually holds. The API has a
# secret named messenger-page-access-token too, but it does not use it, and that one belongs to
# no page Facebook knows: copied blindly, every follow-up came back "(#100) No matching user
# found" on the first live run (6 Oct). A plain value is taken as is; a secretRef is followed.
TOKEN_REF="$(az containerapp show -n "$APP" -g "$RG" --query "properties.template.containers[0].env[?name=='MESSENGER_PAGE_ACCESS_TOKEN'].secretRef|[0]" -o tsv | tr -d '\r')"
if [[ -n "$TOKEN_REF" && "$TOKEN_REF" != "None" ]]; then
  export FJ_TOKEN="$(az containerapp secret show -n "$APP" -g "$RG" --secret-name "$TOKEN_REF" --query value -o tsv | tr -d '\r')"
else
  export FJ_TOKEN="$(az containerapp show -n "$APP" -g "$RG" --query "properties.template.containers[0].env[?name=='MESSENGER_PAGE_ACCESS_TOKEN'].value|[0]" -o tsv | tr -d '\r')"
fi
# Never deploy a token Facebook does not recognise as a page.
curl -sf "https://graph.facebook.com/v21.0/me?fields=id&access_token=$FJ_TOKEN" >/dev/null ||
  { echo "ERROR: the page token read from $APP is not valid for any Facebook page" >&2; exit 1; }
export FJ_ENV_ID="$(az containerapp env show -n "$ENVIRONMENT" -g "$RG" --query id -o tsv | tr -d '\r')"
export FJ_LOCATION="$(az containerapp show -n "$APP" -g "$RG" --query location -o tsv | tr -d '\r')"
export FJ_IMAGE="$IMAGE" FJ_SERVER="$SERVER" FJ_SCHEDULE="$SCHEDULE" FJ_APP="$APP"
# FOLLOWUP_ON=0 deploys it switched off: the schedule runs and does nothing until it is 1.
export FJ_ENABLED="${FOLLOWUP_ON:-1}"
# `export X="$(...)"` does not stop the script when the az call fails, and Azure's API does
# time out now and then - a blank here would deploy a broken job, so stop instead.
for name in FJ_TOKEN FJ_ENV_ID FJ_LOCATION; do
  [[ -n "${!name}" ]] || { echo "ERROR: could not read $name from Azure - run the script again" >&2; exit 1; }
done

# The job is described as a spec file rather than CLI flags: the CLI cannot set the registry
# to the environment's identity (the other jobs got it from Bicep), and it reads "-m" in the
# command as one of its own options. The file holds the page token, so it is removed on exit.
SPEC="$(mktemp --suffix=.yaml)"
trap 'rm -f "$SPEC"' EXIT
az containerapp show -n "$APP" -g "$RG" --query "properties.template.containers[0].env" -o json |
"$PY" -c '
import json, os, sys

skip = {"MESSENGER_PAGE_ACCESS_TOKEN", "MESSENGER_APP_SECRET", "FOLLOWUP_ENABLED", "OUTBOX_ORIGIN", "PYTHONPATH"}
env = [{"name": e["name"], "value": e.get("value") or ""}
       for e in json.load(sys.stdin) if e["name"] not in skip and not e.get("secretRef")]
env += [
    {"name": "MESSENGER_PAGE_ACCESS_TOKEN", "secretRef": "messenger-page-access-token"},
    {"name": "FOLLOWUP_ENABLED", "value": os.environ["FJ_ENABLED"]},
    # A message that arrives mid-follow-up is answered by this job; any team email that turn
    # queues is stamped as the API s, so the API sends it if this job exits first.
    {"name": "OUTBOX_ORIGIN", "value": os.environ["FJ_APP"]},
    # The command runs the file, so the repo root has to be on the path for "import src".
    {"name": "PYTHONPATH", "value": "/app"},
]
spec = {
    "location": os.environ["FJ_LOCATION"],
    "properties": {
        "environmentId": os.environ["FJ_ENV_ID"],
        "configuration": {
            "triggerType": "Schedule",
            "replicaTimeout": 1800,
            "replicaRetryLimit": 0,
            "scheduleTriggerConfig": {
                "cronExpression": os.environ["FJ_SCHEDULE"],
                "parallelism": 1,
                "replicaCompletionCount": 1,
            },
            "registries": [{"server": os.environ["FJ_SERVER"], "identity": "system-environment"}],
            "secrets": [{"name": "messenger-page-access-token", "value": os.environ["FJ_TOKEN"]}],
        },
        "template": {
            "containers": [{
                "name": "followup",
                "image": os.environ["FJ_IMAGE"],
                "command": ["python"],
                "args": ["src/followup/__main__.py"],
                "resources": {"cpu": 0.5, "memory": "1Gi"},
                "env": env,
            }],
        },
    },
}
print(json.dumps(spec, indent=1))
' > "$SPEC"

if az containerapp job show -n "$JOB" -g "$RG" --only-show-errors >/dev/null 2>&1; then
  echo "== updating $JOB"
  az containerapp job update -n "$JOB" -g "$RG" --yaml "$SPEC" --only-show-errors >/dev/null
else
  echo "== creating $JOB"
  az containerapp job create -n "$JOB" -g "$RG" --yaml "$SPEC" --only-show-errors >/dev/null
fi

echo "== done: $JOB runs '$SCHEDULE' on $IMAGE"
echo "   run one now:   az containerapp job start -n $JOB -g $RG"
echo "   see its runs:  az containerapp job execution list -n $JOB -g $RG -o table"
