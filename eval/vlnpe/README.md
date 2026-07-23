# VLN-PE (R2R) 评测

## 使用方法

```bash
# 默认跑 val_seen 第一个 episode
bash eval/vlnpe/run.sh

# 指定 split 和 episode
SPLIT=val_seen EPISODE_INDEX=0 bash eval/vlnpe/run.sh

# 指定 episode_id（精确匹配）
SPLIT=val_unseen EPISODE_ID="some_episode_id" bash eval/vlnpe/run.sh
```

## 数据格式

- 评测文件: `/Data/qiang/InternData-N1/vln_pe/raw_data/r2r/{split}/{split}.json.gz`
- 场景 USD: `/Data/qiang/mp3d_pe/scans/{scene_hash}/matterport_mesh/{hash}/fixed.usd`
- 格式: R2R (Room-to-Room)，`scene_id` 格式为 `mp3d/{hash}/{hash}.glb`

## 输出

- `eval/current_episode.json` — 当前运行的 episode（自动加载）
- `debug/eval/vln_debug.png` — 调试图
- `debug/eval/videos/vln_debug.mp4` — 录像
- `debug/eval/results/<episode>_<time>.json` — 结构化运行结果与失败阶段
- `logs/nav_latest.log` — 日志

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `SPLIT` | `val_seen` | 数据集 split |
| `EPISODE_INDEX` | `0` | episode 序号 |
| `EPISODE_ID` | 空 | episode ID（精确匹配） |
| `VLNPE_DATA_ROOT` | `/Data/qiang/InternData-N1/vln_pe` | 评测数据根目录 |

## 注意

- R2R 数据中 `start_rotation` 为 `[x, y, z, w]` 格式，适配器会自动转为 `[w, x, y, z]`
- 场景 USD 优先选择 `fixed.usd`（而非 `non_metric.usd`）
