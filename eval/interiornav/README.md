# InteriorNav 评测

## 使用方法

```bash
# 默认跑 val_seen 第一个 episode
bash eval/interiornav/run.sh

# 指定 split 和 episode
SPLIT=val_seen EPISODE_INDEX=0 bash eval/interiornav/run.sh

# 指定 episode_id（精确匹配）
SPLIT=val_unseen EPISODE_ID="some_episode_id" bash eval/interiornav/run.sh
```

## 数据格式

- 数据路径: `/Data/qiang/interiornav_data`
- 场景 USD: `raw_data/../scene_data/{scene_id}/{scene_id}.usda`
- 评测文件: `raw_data/{split}/{split}.json.gz`

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
| `INTERIORNAV_DATA_ROOT` | `/Data/qiang/interiornav_data` | 数据根目录 |

`run.sh` 会设置 `ADD_DOME_LIGHT=true`，在场景加载后创建
`/World/defaultDomeLight`。该 Dome 关闭阴影并作为封闭室内的环境补光，
不会修改原始场景 USD。
