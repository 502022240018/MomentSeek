"""Compatibility entrypoint for the versioned Face-group migration."""

from app.maintenance.migrate_face_groups import main


if __name__ == "__main__":
    raise SystemExit(main())
