#!/usr/bin/env bash
# MomentSeek 0829 - 停止 Planner Lab 容器

set -e

CONTAINER_NAME="momentseek-0829-planner-lab"

log() { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date '+%F %T')" "$*"; }

if docker ps -q -f name="^${CONTAINER_NAME}$" >/dev/null 2>&1; then
    log "Stopping Planner Lab container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME"
    log "✅ Planner Lab stopped"
    echo ""
    echo "To start again: ./deploy_planner_lab_0829.sh"
    echo "To remove: docker rm $CONTAINER_NAME"
else
    log "Planner Lab container not running"
fi
