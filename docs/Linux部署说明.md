# Linux 容器部署

## 部署结构

nginx → FastAPI（单进程）→ YOLO / SAM、外部多模态 API；SQLAlchemy + SQLite 保存分析原图、完整双路结果和备注。图片以编码形式随记录保存在数据库中，删除记录会一并删除该记录的图像内容。数据库文件空间可能不会立即缩小。

当前使用 Ubuntu 22.04 镜像中的 Python 3.10，Windows 基线为 Python 3.12；依赖和模型必须在实际容器中验证。Python 官方镜像若在当前镜像加速器不可获取，不代表上游不存在。

## 准备

安装 Docker / Compose，确保基础镜像、Ubuntu 软件源、Python 包索引与多模态 API 可达。复制代码和 `deploy/`、Dockerfile、compose.yaml、requirements-server.txt。

准备 `.env`（参考 `.env.example`，勿提交密钥）、`models/yolo_runs/baseline_10epochs/weights/best.pt` 和 `models/sam2.1_t.pt`。配置中的模型路径不能指向 Windows 路径。

```bash
mkdir -p storage
chown 10001:10001 storage
chmod 600 .env
docker compose config --quiet
docker compose build app
docker compose up -d
docker compose ps
curl http://127.0.0.1:8080/ready
```

容器内应用以普通用户运行。`storage/` 挂载保存数据库；`models/` 只读挂载。构建白名单仅包含应用源码与运行依赖，不包含 .env、权重、数据集和 SSH 密钥。

## 浏览器访问

默认 nginx 仅绑定服务器回环地址 8080。页面和历史记录尚无多用户认证，使用 SSH 隧道保护访问，公网不直接开放应用端口：

```powershell
ssh -N -L 8080:127.0.0.1:8080 用户名@服务器公网IP
```

保持窗口连接，在本机浏览器访问 `http://127.0.0.1:8080`。该方式可完成真实服务器演示。若要让审核者直接从公网访问，先落实 HTTPS 和访问认证，再调整端口和安全组；不直接公开未认证的历史接口。

## 运行、更新、排错

```bash
docker compose logs --tail=100 app nginx
docker compose restart
docker compose up -d --build
docker compose stop
```

`/health` 检查进程响应，`/ready` 检查权重文件和密钥配置存在，不证明模型可加载或外部 API 可用。完整验收需要上传真实图片，检查两路结果与 SAM。

备份先停止 app 后复制 `storage/`，完成后重新启动；恢复时保留目录的 10001 用户权限。停止或删除容器不会自动停止云服务器等资源计费。

历史页面支持分页、详情、备注修改和删除。单机低并发演示为当前范围；应定期清理不需要的历史图片。nginx 对分析请求限流，单次上传文件最大 10 MB；请求整体上限 11 MB。

## 验收记录

2026 年 10 月 17 日，在 Ubuntu 22.04 x86_64、4 vCPU / 8 GiB 的 ECS 上通过以下验证：

- Docker Compose 构建并启动 app、nginx，应用健康检查通过，`pip check` 无依赖冲突。
- 同图真实 YOLO 与多模态 API 调用成功，浏览器展示双路结果；SAM 点提示与网页框提示分割成功。
- 历史记录增删查改通过，应用重启及容器重建后原图、结果和备注保留。
- 损坏图片返回 400、超过 10 MB 文件返回 413、无效分割坐标返回 400。
- 本地可执行 `python scripts/check_history.py` 检查独立临时数据库的 CRUD 与重连保留；`python scripts/check_stage3_baseline.py 图片路径` 检查模型和上传契约，添加 `--remote` 才发起真实收费 API 调用。

服务器实际依赖快照见 `requirements-server.lock.txt`（Python 3.10，包含 CPU 版 torch / torchvision）；Dockerfile 从官方 CPU 索引安装模型运行包。快照用于核对实测环境，重建命令以 Dockerfile 为准。单图成功不代表所有类别准确率或并发性能保证。
