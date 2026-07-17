#!/usr/bin/env bash
set -uo pipefail

IMAGE="${IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:3.0.0b2-800I-A2-py311-openeuler24.03-lts}"
NPU_ID="${NPU_ID:-5}"

title() { printf '\n========== %s ==========\n' "$1"; }
run() {
  printf '\n$ %s\n' "$*"
  "$@"
  local rc=$?
  printf 'exit_code=%s\n' "$rc"
  return 0
}

title "1. Host network (control)"
run curl -ILsS --connect-timeout 5 --max-time 15 -o /dev/null \
  -w 'pypi http=%{http_code} ip=%{remote_ip} time=%{time_total}s error=%{errormsg}\n' \
  https://pypi.org/simple/

title "2. Container default bridge network"
run docker run --rm "$IMAGE" bash -lc \
  "curl -ILsS --connect-timeout 5 --max-time 15 -o /dev/null -w 'bridge http=%{http_code} ip=%{remote_ip} time=%{time_total}s error=%{errormsg}\\n' https://pypi.org/simple/"

title "3. Container host network"
run docker run --rm --network host "$IMAGE" bash -lc \
  "curl -ILsS --connect-timeout 5 --max-time 15 -o /dev/null -w 'hostnet http=%{http_code} ip=%{remote_ip} time=%{time_total}s error=%{errormsg}\\n' https://pypi.org/simple/"
run docker run --rm --network host "$IMAGE" bash -lc \
  "curl -ILsS --connect-timeout 5 --max-time 15 -o /dev/null -w 'modelscope http=%{http_code} ip=%{remote_ip} time=%{time_total}s error=%{errormsg}\\n' https://modelscope.cn/"

COMMON=(
  --rm
  --device "/dev/davinci${NPU_ID}"
  --device /dev/davinci_manager
  --device /dev/devmm_svm
  --device /dev/hisi_hdc
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0
  -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/cann-8.5.1/lib64
)

title "4. Ascend driver library on host and in container"
run bash -lc 'find /usr/local/Ascend/driver -name "libascend_hal.so*" -o -name "libascendcl.so*" 2>/dev/null | sort | head -30'
run docker run "${COMMON[@]}" "$IMAGE" bash -lc \
  'find /usr/local/Ascend/driver -name "libascend_hal.so*" -o -name "libascendcl.so*" 2>/dev/null | sort | head -30; ldconfig -p 2>/dev/null | grep -E "libascend_(hal|cl)" || true'

title "5. Clean torch + torch_npu test on physical NPU ${NPU_ID}"
run docker run "${COMMON[@]}" "$IMAGE" python3 -c \
  'import torch; import torch_npu; x=torch.arange(8,dtype=torch.float32,device="npu:0"); print("torch",torch.__version__); print("torch_npu",torch_npu.__version__); print("available",torch.npu.is_available(),"count",torch.npu.device_count()); print("result",(x*2).cpu().tolist())'

title "6. Each import in a fresh process"
run docker run "${COMMON[@]}" "$IMAGE" python3 -c \
  'import torch; import torch_npu; import torchvision; print("torchvision",torchvision.__version__)'
run docker run "${COMMON[@]}" "$IMAGE" python3 -c \
  'import torch; import torch_npu; import transformers; from transformers import Siglip2Model, AutoProcessor; print("transformers",transformers.__version__,"SigLIP2 import PASS")'

title "7. Installed versions and Ascend runtime inside image"
run docker run "${COMMON[@]}" "$IMAGE" bash -lc \
  'python3 -m pip show torch torch-npu torchvision transformers 2>/dev/null | grep -E "^(Name|Version):"; find /usr/local/Ascend -maxdepth 4 \( -name version.cfg -o -name version.info \) -type f -print 2>/dev/null | sort'

title "Summary hint"
printf '%s\n' \
  'Expected: bridge may fail, while --network host should return HTTP 200.' \
  'Paste the complete output back. This script installs nothing and creates no persistent container.'
