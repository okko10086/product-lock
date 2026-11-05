# 生成器 CLI 契约

product-lock 不绑定任何生成服务。它只认一个命令行约定：你写一个 CLI，把 `image` / `video`
两个动词转成你自己的 API 调用即可。

## 约定

```
<cli> image  --image PATH --prompt TEXT --count N --ratio R --size S --model NAME
<cli> video  --image PATH --prompt TEXT --duration SEC --ratio W:H --model NAME
<cli> fetch  --thread ID --run ID --outdir DIR
<cli> balance
```

| 动词 | 花钱 | 必须输出 |
|---|---|---|
| `image` | 是 | 一行 JSON：`{"thread_id": "...", "run_id": "..."}` |
| `video` | 是 | 一行 JSON：`{"thread_id": "...", "run_id": "..."}` |
| `fetch` | 否 | 完成后输出 `{"completed": true}`，文件写进 `--outdir` |
| `balance` | 否 | 一行 JSON：`{"credits": 1234}`（可自行扩展） |

其它动词（例如查询历史、取消任务）随便加，守门人不拦，原样透传给真 CLI。

## 只输出 JSON

提交类动词的 stdout **最后一行必须是 JSON**，插件按行解析：

```bash
$ my-generator video --image /tmp/a.png --prompt "..." --duration 4 --ratio 3:4
{"thread_id": "4f2a", "run_id": "9c1b"}
```

进度、日志、人看的提示一律走 stderr，否则会把解析搞坏。

## 接进来

```bash
export GENERATOR_CLI="$PWD/tools/guard"          # 插件指向守门人
export GENERATOR_CLI_REAL=/path/to/my-generator  # 守门人指向真 CLI
chmod +x tools/guard /path/to/my-generator
```

`tools/guard` 会在 `video` / `image` 前面查一次性令牌。你的 CLI 不需要知道令牌存在——
守门人核销之后就把它从环境里摘掉了。

## 最小适配器

`examples/mock_generator.sh` 是一个能跑通全流程的假生成器，不花一分钱，用来验证接线：

```bash
export GENERATOR_CLI="$PWD/tools/guard"
export GENERATOR_CLI_REAL="$PWD/examples/mock_generator.sh"

python3 core/plugin.py plan --model out/model.json --script "test" --out plan.json
# 给 plan.json 补一个 "first_frame_image"
python3 core/plugin.py approve --plan plan.json
python3 core/plugin.py run --plan plan.json --outdir out/videos --confirm
```

## 换别的厂商

改三处就够：

1. 写一个新的 `<cli>`，实现上面四个动词
2. `export GENERATOR_CLI_REAL=<你的 cli>`
3. 如果花钱的动词不叫 `image` / `video`，用 `GUARD_SPEND_VERBS` 覆盖：

```bash
export GUARD_SPEND_VERBS=video,image,render,train
```

不需要改 `core/plugin.py` 一行代码。
