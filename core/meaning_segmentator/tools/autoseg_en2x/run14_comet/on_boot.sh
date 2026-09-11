#!/bin/bash
# 재부팅 직후 한 번만: run14 버전별 COMET 체인을 tmux 로 띄운다. 띄운 뒤 자기 crontab 항목을 지운다.
# 등록: crontab -l 에 `@reboot /home/mobility/STiTy/core/meaning_segmentator/tools/autoseg_en2x/run14_comet/on_boot.sh`
export HOME=/home/mobility
export PATH=/home/mobility/.local/bin:/home/mobility/miniforge3/envs/tools/bin:/usr/local/bin:/usr/bin:/bin
REPO=/home/mobility/STiTy
LOG=$REPO/core/meaning_segmentator/experiment/artifacts/en2x/logs/run14_comet.launch.log
sleep 90                                   # 드라이버·파일시스템이 자리 잡을 시간
echo "[$(date '+%F %T')] boot: launching run14_comet chain" >> "$LOG"
cd "$REPO" || exit 1
if ! tmux has-session -t run14_comet 2>/dev/null; then
  tmux new-session -d -s run14_comet -c "$REPO" \
    "bash core/meaning_segmentator/tools/autoseg_en2x/run14_comet/chain.sh"
  echo "[$(date '+%F %T')] tmux run14_comet started" >> "$LOG"
else
  echo "[$(date '+%F %T')] tmux run14_comet already running" >> "$LOG"
fi
# 한 번만 돌게 crontab 에서 자기 줄을 뺀다
crontab -l 2>/dev/null | grep -v "run14_comet/on_boot.sh" | crontab -
echo "[$(date '+%F %T')] crontab entry removed" >> "$LOG"
