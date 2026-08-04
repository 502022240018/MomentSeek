from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
from pathlib import Path


REQUIRED = (
    "APP_IMAGE",
    "APP_CONTAINER_NAME",
    "COMPOSE_PROJECT_NAME",
    "MOMENTSEEK_NETWORK_NAME",
    "APP_PORT",
    "APP_PUBLIC_URL",
    "HOST_RUNTIME_DIR",
    "HOST_MODEL_DIR",
    "HOST_NPU_DEVICE_ID",
)


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def run(*command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def compose_command() -> tuple[str, ...] | None:
    if shutil.which("docker") and run("docker", "compose", "version").returncode == 0:
        return ("docker", "compose")
    if shutil.which("docker-compose") and run("docker-compose", "version").returncode == 0:
        return ("docker-compose",)
    return None


def port_available(port: int) -> bool:
    sock = socket.socket()
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="MomentSeek Ascend 部署前只读检查")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--with-milvus",
        action="store_true",
        help="同时检查本仓库自带的 Milvus/etcd/MinIO 镜像和凭据",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="升级已有同名实例；允许容器名和应用端口已被该实例占用",
    )
    args = parser.parse_args()

    errors: list[str] = []
    if not args.env_file.is_file():
        print(f"[ERROR] 配置文件不存在: {args.env_file}")
        return 1
    env = read_env(args.env_file)

    for key in REQUIRED:
        if not env.get(key):
            errors.append(f"缺少配置 {key}")
    if "SERVER_IP" in env.get("APP_PUBLIC_URL", ""):
        errors.append("APP_PUBLIC_URL 仍包含模板值 SERVER_IP")
    if args.with_milvus:
        for key in ("MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD"):
            if not env.get(key) or env.get(key, "").startswith("CHANGE_ME"):
                errors.append(f"{key} 未设置或仍是模板值")

    if shutil.which("docker") is None:
        errors.append("找不到 docker")
    else:
        if compose_command() is None:
            errors.append("docker compose 和 docker-compose 均不可用")
        image = env.get("APP_IMAGE")
        if image and run("docker", "image", "inspect", image).returncode:
            errors.append(f"本机不存在应用镜像 {image}")
        if args.with_milvus:
            for key in ("MILVUS_IMAGE", "ETCD_IMAGE", "MINIO_IMAGE"):
                image = env.get(key)
                if not image or run("docker", "image", "inspect", image).returncode:
                    errors.append(f"本机不存在 {key}={image or '<未设置>'}")
        name = env.get("APP_CONTAINER_NAME")
        if (
            name
            and not args.upgrade
            and run("docker", "container", "inspect", name).returncode == 0
        ):
            errors.append(f"容器名已存在 {name}")

    model_dir = Path(env.get("HOST_MODEL_DIR", "/path/does/not/exist"))
    if not model_dir.is_dir():
        errors.append(f"模型目录不存在 {model_dir}")

    runtime_dir = Path(env.get("HOST_RUNTIME_DIR", "/path/does/not/exist"))
    if runtime_dir.exists() and not runtime_dir.is_dir():
        errors.append(f"运行路径不是目录 {runtime_dir}")
    elif runtime_dir.exists() and not os.access(runtime_dir, os.W_OK):
        errors.append(f"运行目录不可写 {runtime_dir}")

    physical = env.get("HOST_NPU_DEVICE_ID", "")
    if physical and not Path(f"/dev/davinci{physical}").exists():
        errors.append(f"Ascend 设备不存在 /dev/davinci{physical}")
    for device in ("/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc"):
        if not Path(device).exists():
            errors.append(f"Ascend 管理设备不存在 {device}")

    try:
        port = int(env.get("APP_PORT", ""))
    except ValueError:
        errors.append("APP_PORT 不是整数")
    else:
        if not 1 <= port <= 65535:
            errors.append("APP_PORT 超出 1-65535")
        elif not args.upgrade and not port_available(port):
            errors.append(f"端口已被占用 {port}")

    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        print("[ERROR] 预检未通过，未对服务器做任何修改。")
        return 1
    print("[OK] Docker、应用镜像、端口、目录和 Ascend 设备预检通过。")
    print(f"[OK] Compose 命令: {' '.join(compose_command() or ())}")
    print("[WARN] 本脚本不能判断 NPU 是否被其他进程使用，请人工核对 npu-smi info。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
