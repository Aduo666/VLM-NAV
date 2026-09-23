# InternVLA-N1 System 2 服务

该目录提供远程 S2 服务端和本地接口测试脚本。服务端使用三个独立接口：

```text
POST /infer_once   无历史单张图片测试
POST /reset        创建或清空导航会话
POST /infer        使用已有会话连续推断
```

## 启动服务器

将 `s2_server.py` 和 `requirements_s2_server.txt` 放到模型服务器，在 InternVLA 环境运行：

```bash
python s2_server.py \
  --model-path /path/to/InternVLA-N1-S2 \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5801
```

健康检查：

```bash
curl http://127.0.0.1:5801/health
```

## 单张图片测试

`once` 模式不会保存图像历史：

```bash
python InternVLA-N1-S2/test_s2_client.py \
  --server http://服务器IP:5801 \
  --mode once \
  --image /path/to/front.jpg \
  --instruction "Leave the kitchen and enter the bedroom."
```

## 连续会话测试

第一张前视图创建会话：

```bash
python InternVLA-N1-S2/test_s2_client.py \
  --server http://服务器IP:5801 \
  --mode session \
  --reset \
  --session-id demo-001 \
  --request-id 1 \
  --view front \
  --image /path/to/front.jpg \
  --instruction "Leave the kitchen and enter the bedroom."
```

如果返回 `discrete_action: [5]`，使用相同会话上传下视图：

```bash
python InternVLA-N1-S2/test_s2_client.py \
  --server http://服务器IP:5801 \
  --mode session \
  --session-id demo-001 \
  --request-id 2 \
  --view down \
  --look-down \
  --image /path/to/down.jpg \
  --instruction "Leave the kitchen and enter the bedroom."
```

服务端对外返回官方字段：

```json
{"pixel_goal": [120, 220], "pixel_space_hw": [384, 384]}
```

或：

```json
{"discrete_action": [5]}
```
