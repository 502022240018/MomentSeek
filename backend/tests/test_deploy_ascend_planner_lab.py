from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "deploy_ascend_planner_lab.sh"
)


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_planner_lab_visual_ann_override_uses_real_settings_key_after_env_file():
    script = _script()

    assert (
        'PLANNER_LAB_VISUAL_ANN_TOP_K="${PLANNER_LAB_VISUAL_ANN_TOP_K:-'
        '${VISUAL_ANN_TOP_K:-2000}}"'
    ) in script
    env_file = script.index('--env-file "$ENV_FILE"')
    override = script.index(
        '-e "VISUAL_ANN_TOP_K=$PLANNER_LAB_VISUAL_ANN_TOP_K"'
    )
    assert override > env_file
    assert "PLANNER_LAB_VISUAL_ANN_TOP_K < 1" in script
    assert "PLANNER_LAB_VISUAL_ANN_TOP_K > 16383" in script


def test_planner_lab_deployment_overrides_stale_release_metadata():
    script = _script()
    env_file = script.index('--env-file "$ENV_FILE"')

    for assignment in (
        '-e "RELEASE_ID=$PLANNER_LAB_RELEASE_ID"',
        '-e "GIT_COMMIT=$PLANNER_LAB_GIT_COMMIT"',
        '-e "IMAGE_TAG=$IMAGE_TAG"',
    ):
        assert script.index(assignment) > env_file

    assert 'git -C "$SOURCE_DIR" rev-parse --verify HEAD' in script
    assert '"release_id", "git_commit", "image_tag"' in script


def test_planner_lab_deployment_does_not_make_1000_the_global_default():
    script = _script()

    assert "VISUAL_ANN_TOP_K:-1000" not in script
    assert "PLANNER_LAB_VISUAL_ANN_TOP_K:-1000" not in script


def test_planner_lab_deployment_restarts_original_if_backup_rename_fails():
    script = _script()

    rename_guard = script.index(
        'if ! docker rename "$CONTAINER_NAME" "$BACKUP_NAME"; then'
    )
    restart = script.index(
        'docker start "$CONTAINER_NAME" >/dev/null || true', rename_guard
    )
    guard_end = script.index("  fi", restart)

    assert rename_guard < restart < guard_end
