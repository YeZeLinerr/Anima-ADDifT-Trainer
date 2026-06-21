"""Small local WebUI launcher for Anima ADDifT training.

This intentionally uses only Python's standard library. The server itself and
the training subprocess run with the portable venv selected by the BAT file.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_DIR / "anima_train_addift.py"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class TrainingState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.process: subprocess.Popen[str] | None = None
        self.logs: deque[dict[str, Any]] = deque(maxlen=5000)
        self.next_log_id = 1
        self.command = ""
        self.started_at: float | None = None
        self.exit_code: int | None = None
        self.status = "idle"

    def log(self, text: str, stream: str = "system") -> None:
        text = text.rstrip("\r\n")
        if not text:
            return
        with self.lock:
            self.logs.append(
                {
                    "id": self.next_log_id,
                    "time": time.strftime("%H:%M:%S"),
                    "stream": stream,
                    "text": text,
                }
            )
            self.next_log_id += 1

    def snapshot(self, after: int = 0) -> dict[str, Any]:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            return {
                "status": "running" if running else self.status,
                "running": running,
                "pid": self.process.pid if running and self.process else None,
                "exit_code": self.exit_code,
                "started_at": self.started_at,
                "elapsed": round(time.time() - self.started_at, 1) if running and self.started_at else None,
                "command": self.command,
                "logs": [entry for entry in self.logs if entry["id"] > after],
            }


STATE = TrainingState()


def _string(value: Any) -> str:
    return str(value or "").strip()


def _required_file(data: dict[str, Any], key: str, label: str) -> Path:
    path = Path(_string(data.get(key))).expanduser()
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a file / 不存在或不是文件: {path}")
    return path.resolve()


def _required_dir(data: dict[str, Any], key: str, label: str, create: bool = False) -> Path:
    path = Path(_string(data.get(key))).expanduser()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"{label} does not exist or is not a directory / 不存在或不是目录: {path}")
    return path.resolve()


def _image_stems(directory: Path) -> set[str]:
    return {
        item.stem.casefold()
        for item in directory.iterdir()
        if item.is_file() and item.suffix.casefold() in IMAGE_EXTENSIONS
    }


def _toml_string(path: Path | str) -> str:
    return json.dumps(str(path), ensure_ascii=False)


def build_training(data: dict[str, Any]) -> tuple[list[str], str, int]:
    model = _required_file(data, "model", "Anima DiT")
    qwen3 = _required_file(data, "qwen3", "Qwen3")
    vae = _required_file(data, "vae", "Anima VAE")
    target_dir = _required_dir(data, "target_dir", "Target image directory / Target 图片目录")
    source_dir = _required_dir(data, "source_dir", "Source image directory / Source 图片目录")
    output_dir = _required_dir(data, "output_dir", "Output directory / 输出目录", create=True)

    target_stems = _image_stems(target_dir)
    source_stems = _image_stems(source_dir)
    if not target_stems:
        raise ValueError("No supported images in Target directory / Target 目录中没有支持的图片")
    if not source_stems:
        raise ValueError("No supported images in Source directory / Source 目录中没有支持的图片")
    missing_source = sorted(target_stems - source_stems)
    missing_target = sorted(source_stems - target_stems)
    if missing_source or missing_target:
        details = []
        if missing_source:
            details.append("Missing from Source / Source 缺少: " + ", ".join(missing_source[:10]))
        if missing_target:
            details.append("Missing from Target / Target 缺少: " + ", ".join(missing_target[:10]))
        raise ValueError("Image filename stems are not paired / 图片文件名 stem 未一一配对; " + "; ".join(details))

    output_name = _string(data.get("output_name")) or "anima_addift"
    if any(char in output_name for char in '<>:"/\\|?*'):
        raise ValueError("Output name contains reserved Windows characters / 输出名称包含 Windows 保留字符")

    resolution = int(data.get("resolution", 512))
    repeats = int(data.get("repeats", 100))
    steps = int(data.get("steps", 100))
    dim = int(data.get("network_dim", 8))
    alpha = float(data.get("network_alpha", dim))
    learning_rate = float(data.get("learning_rate", 5e-5))
    save_every = int(data.get("save_every", 100))
    min_timestep = int(data.get("paired_min_timestep", 500))
    max_timestep = int(data.get("paired_max_timestep", 1000))
    if min(resolution, repeats, steps, dim) <= 0:
        raise ValueError("Resolution, repeats, steps, and Network Dim must be positive / 分辨率、重复次数、训练步数和 Network Dim 必须大于 0")
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive / 学习率必须大于 0")
    if not 0 <= min_timestep < max_timestep <= 1000:
        raise ValueError("Timestep range must satisfy 0 ≤ min < max ≤ 1000 / Timestep 范围必须满足 0 ≤ 最小值 < 最大值 ≤ 1000")

    work_dir = output_dir / ".addift_webui"
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = work_dir / f"{output_name}_dataset.toml"
    caption_extension = _string(data.get("caption_extension")) or ".txt"
    caption_prefix = _string(data.get("caption_prefix"))
    dataset_text = (
        "[[datasets]]\n"
        f"batch_size = {int(data.get('batch_size', 1))}\n"
        f"resolution = [{resolution}, {resolution}]\n"
        "enable_bucket = true\n"
        "bucket_no_upscale = true\n\n"
        f"max_bucket_reso = {max(1024, resolution)}\n"
        "bucket_reso_steps = 64\n\n"
        "  [[datasets.subsets]]\n"
        f"  image_dir = {_toml_string(target_dir)}\n"
        f"  conditioning_data_dir = {_toml_string(source_dir)}\n"
        f"  num_repeats = {repeats}\n"
        f"  caption_extension = {_toml_string(caption_extension)}\n"
        + (f"  caption_prefix = {_toml_string(caption_prefix)}\n" if caption_prefix else "")
    )
    dataset_path.write_text(dataset_text, encoding="utf-8")

    mixed_precision = _string(data.get("mixed_precision")) or "bf16"
    optimizer = _string(data.get("optimizer")) or "AdamW8bit"
    args = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_cpu_threads_per_process",
        "1",
        "--mixed_precision",
        mixed_precision,
        str(TRAIN_SCRIPT),
        f"--dataset_config={dataset_path}",
        f"--dit_path={model}",
        f"--qwen3_path={qwen3}",
        f"--vae_path={vae}",
        f"--output_dir={output_dir}",
        f"--output_name={output_name}",
        "--save_model_as=safetensors",
        "--network_module=networks.lora_anima",
        f"--network_dim={dim}",
        f"--network_alpha={alpha:g}",
        "--network_train_unet_only",
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        f"--learning_rate={learning_rate:g}",
        f"--optimizer_type={optimizer}",
        "--lr_scheduler=constant",
        f"--max_train_steps={steps}",
        f"--mixed_precision={mixed_precision}",
        "--gradient_checkpointing",
        "--max_data_loader_n_workers=0",
        f"--vae_batch_size={int(data.get('vae_batch_size', 1))}",
        f"--paired_slider_scale={float(data.get('paired_slider_scale', 0.25)):g}",
        f"--paired_min_timestep={min_timestep}",
        f"--paired_max_timestep={max_timestep}",
        f"--paired_mask_threshold={float(data.get('paired_mask_threshold', 1.0)):g}",
        f"--paired_background_weight={float(data.get('paired_background_weight', 0.1)):g}",
    ]
    if 0 < save_every <= steps:
        args.append(f"--save_every_n_steps={save_every}")
    args.append("--paired_difference_mask" if bool(data.get("paired_difference_mask", True)) else "--no-paired_difference_mask")
    args.append("--paired_mask_normalize" if bool(data.get("paired_mask_normalize", True)) else "--no-paired_mask_normalize")
    extra = _string(data.get("extra_args"))
    if extra:
        parsed = shlex.split(extra, posix=False)
        args.extend(token[1:-1] if len(token) >= 2 and token[0] == token[-1] == '"' else token for token in parsed)

    return args, str(dataset_path), len(target_stems)


def _reader(process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    try:
        for line in iter(process.stdout.readline, ""):
            STATE.log(line, "train")
    except Exception:
        STATE.log(traceback.format_exc(), "error")
    finally:
        code = process.wait()
        with STATE.lock:
            STATE.exit_code = code
            STATE.status = "finished" if code == 0 else "failed"
        STATE.log(f"Training process ended / 训练进程已结束. Exit code / 退出码: {code}", "system")


def start_training(data: dict[str, Any]) -> dict[str, Any]:
    with STATE.lock:
        if STATE.process is not None and STATE.process.poll() is None:
            raise ValueError("Training is already running / 已有训练正在运行")

    args, dataset_path, pair_count = build_training(data)
    command = subprocess.list2cmdline(args)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        args,
        cwd=PROJECT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    with STATE.lock:
        STATE.process = process
        STATE.logs.clear()
        STATE.next_log_id = 1
        STATE.command = command
        STATE.started_at = time.time()
        STATE.exit_code = None
        STATE.status = "running"
    STATE.log(f"Dataset config generated / 已生成数据集配置: {dataset_path}")
    STATE.log(f"Validated Source/Target pairs / 已验证 Source/Target 图片: {pair_count}")
    STATE.log("Launch command / 启动命令: " + command)
    threading.Thread(target=_reader, args=(process,), daemon=True).start()
    return {"ok": True, "pid": process.pid, "dataset": dataset_path, "pairs": pair_count, "command": command}


def stop_training() -> dict[str, Any]:
    with STATE.lock:
        process = STATE.process
        if process is None or process.poll() is not None:
            return {"ok": True, "message": "No training is running / 没有正在运行的训练"}
        pid = process.pid
    STATE.log(f"Stopping training process tree / 正在停止训练进程树 PID {pid}...")
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, check=False)
    else:
        process.terminate()
    return {"ok": True, "message": f"Stop request sent / 已发送停止请求: PID {pid}"}


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Anima ADDifT Trainer</title>
<style>
:root{color-scheme:dark;--bg:#0b0d12;--panel:#141821;--line:#293142;--text:#e8ecf4;--muted:#98a2b3;--accent:#8b5cf6;--accent2:#22c55e;--danger:#ef4444}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#20153a 0,transparent 32%),var(--bg);color:var(--text);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}
.wrap{max-width:1280px;margin:auto;padding:28px}.head{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:20px}h1{margin:0;font-size:28px}.sub{color:var(--muted)}
.layout{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(380px,.9fr);gap:18px}.panel{background:rgba(20,24,33,.94);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 18px 60px #0005}
h2{font-size:16px;margin:0 0 14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.full{grid-column:1/-1}label{display:block;color:#cbd5e1;font-weight:600;margin-bottom:5px}
input,select,textarea{width:100%;border:1px solid #364157;border-radius:8px;background:#0d1119;color:var(--text);padding:9px 10px;outline:none}input:focus,select:focus,textarea:focus{border-color:var(--accent)}
input[type=checkbox]{width:auto;accent-color:var(--accent)}.check{display:flex;gap:8px;align-items:center;padding-top:24px}.check label{margin:0}.hint{font-size:12px;color:var(--muted);margin-top:4px}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}.btn{border:0;border-radius:9px;padding:10px 16px;font-weight:700;cursor:pointer;color:white;background:var(--accent)}.btn.secondary{background:#344054}.btn.stop{background:var(--danger)}.btn:disabled{opacity:.45;cursor:not-allowed}
.head-actions{display:flex;align-items:center;gap:12px}.lang{border:1px solid #4b5870;border-radius:8px;background:#1d2431;color:var(--text);padding:7px 11px;cursor:pointer;font-weight:700}.status{display:flex;align-items:center;gap:8px;font-weight:700}.dot{width:9px;height:9px;border-radius:50%;background:#64748b}.dot.running{background:var(--accent2);box-shadow:0 0 12px var(--accent2)}
pre{height:650px;overflow:auto;white-space:pre-wrap;word-break:break-word;background:#080a0f;border:1px solid #252b38;border-radius:10px;padding:12px;margin:12px 0 0;font:12px/1.55 Consolas,monospace;color:#d9e2f2}
.bar{display:flex;justify-content:space-between;gap:12px;align-items:center}.tiny{font-size:12px;color:var(--muted)}details{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;color:#c4b5fd;font-weight:700}
@media(max-width:900px){.layout{grid-template-columns:1fr}.wrap{padding:16px}pre{height:420px}} 
</style>
</head>
<body><div class="wrap">
<div class="head"><div><h1>Anima ADDifT Trainer</h1><div class="sub" data-i18n="subtitle"></div></div><div class="head-actions"><button id="langToggle" class="lang" type="button"></button><div class="status"><span id="dot" class="dot"></span><span id="status"></span></div></div></div>
<div class="layout">
<section class="panel"><h2 data-i18n="modelsDataset"></h2><form id="form"><div class="grid">
<div class="full"><label>Anima DiT (.safetensors)</label><input name="model" required placeholder="D:\models\anima.safetensors"></div>
<div class="full"><label>Qwen3 Text Encoder</label><input name="qwen3" required placeholder="D:\models\qwen3.safetensors"></div>
<div class="full"><label>Anima VAE</label><input name="vae" required placeholder="D:\models\anima_vae.safetensors"></div>
<div><label data-i18n="sourceImages"></label><input name="source_dir" required></div>
<div><label data-i18n="targetImages"></label><input name="target_dir" required></div>
<div class="full hint" data-i18n="pairHint"></div>
<div><label data-i18n="outputDirectory"></label><input name="output_dir" required></div>
<div><label data-i18n="outputName"></label><input name="output_name" value="anima_addift"></div>
</div>
<details open><summary data-i18n="basicTraining"></summary><div class="grid" style="margin-top:12px">
<div><label data-i18n="resolution"></label><select name="resolution"><option>512</option><option>768</option><option>1024</option><option>1536</option></select></div>
<div><label data-i18n="imageRepeats"></label><input name="repeats" type="number" min="1" value="100"></div>
<div><label data-i18n="trainingSteps"></label><input name="steps" type="number" min="1" value="100"></div>
<div><label data-i18n="saveEvery"></label><input name="save_every" type="number" min="0" value="100"></div>
<div><label>Network Dim</label><input name="network_dim" type="number" min="1" value="8"></div>
<div><label>Network Alpha</label><input name="network_alpha" type="number" min="0" step="0.1" value="8"></div>
<div><label data-i18n="learningRate"></label><input name="learning_rate" value="5e-5"></div>
<div><label data-i18n="optimizer"></label><select name="optimizer"><option>AdamW8bit</option><option>AdamW</option><option>Adafactor</option></select></div>
<div><label data-i18n="precision"></label><select name="mixed_precision"><option>bf16</option><option>fp16</option><option>no</option></select></div>
<div><label>Batch Size</label><input name="batch_size" type="number" min="1" value="1"></div>
<div><label data-i18n="sliderScale"></label><input name="paired_slider_scale" type="number" min="0.01" step="0.05" value="0.25"></div>
<div><label data-i18n="backgroundWeight"></label><input name="paired_background_weight" type="number" min="0" max="1" step="0.05" value="0.1"></div>
<div><label data-i18n="minimumTimestep"></label><input name="paired_min_timestep" type="number" min="0" max="999" step="10" value="500"></div>
<div><label data-i18n="maximumTimestep"></label><input name="paired_max_timestep" type="number" min="1" max="1000" step="10" value="1000"></div>
<div class="full hint" data-i18n="timestepHint"></div>
</div></details>
<details><summary data-i18n="advanced"></summary><div class="grid" style="margin-top:12px">
<div><label>VAE Batch Size</label><input name="vae_batch_size" type="number" min="1" value="1"></div>
<div><label data-i18n="captionExtension"></label><input name="caption_extension" value=".txt"></div>
<div class="full"><label data-i18n="captionPrefix"></label><input name="caption_prefix" data-i18n-placeholder="captionPrefixPlaceholder"></div>
<div class="full hint" data-i18n="captionHint"></div>
<div class="check"><input id="paired_difference_mask" name="paired_difference_mask" type="checkbox" checked><label for="paired_difference_mask" data-i18n="differenceMask"></label></div>
<div class="check"><input id="paired_mask_normalize" name="paired_mask_normalize" type="checkbox" checked><label for="paired_mask_normalize" data-i18n="normalizeMask"></label></div>
<div><label>Mask Threshold</label><input name="paired_mask_threshold" type="number" min="0.01" step="0.1" value="1.0"></div>
<div class="hint" style="padding-top:28px" data-i18n="directionHint"></div>
<div class="full"><label data-i18n="extraArguments"></label><textarea name="extra_args" rows="3" data-i18n-placeholder="extraArgumentsPlaceholder"></textarea></div>
</div></details>
<div class="actions"><button class="btn" id="start" type="submit" data-i18n="startTraining"></button><button class="btn stop" id="stop" type="button" disabled data-i18n="stopTraining"></button><button class="btn secondary" id="saveConfig" type="button" data-i18n="saveConfig"></button><button class="btn secondary" id="loadConfig" type="button" data-i18n="loadConfig"></button><input id="configFile" type="file" accept=".json,application/json" hidden></div>
</form></section>
<section class="panel"><div class="bar"><h2 style="margin:0" data-i18n="trainingLog"></h2><span id="meta" class="tiny"></span></div><pre id="log"></pre></section>
</div></div>
<script>
const messages={
en:{subtitle:'Source → Target Paired Difference LoRA · Local Port 3001',modelsDataset:'Models & Dataset',sourceImages:'Source Images (Before)',targetImages:'Target Images (After)',pairHint:'Pair images by filename stem, for example source/001.png ↔ target/001.png.',outputDirectory:'Output Directory',outputName:'Output Name',basicTraining:'Basic Training',resolution:'Resolution',imageRepeats:'Image Repeats',trainingSteps:'Training Steps',saveEvery:'Save Every N Steps',learningRate:'Learning Rate',optimizer:'Optimizer',precision:'Precision',sliderScale:'Training Slider Scale',backgroundWeight:'Background Loss Weight',minimumTimestep:'Minimum Timestep',maximumTimestep:'Maximum Timestep',timestepHint:'Local edits and pose: 500–1000; color and style: start around 200–400.',advanced:'Advanced',captionExtension:'Caption Extension',captionPrefix:'Caption Prefix',captionPrefixPlaceholder:'Example: my_edit_trigger',captionHint:'Prepended to each caption; also works as a shared prompt when caption files are absent.',differenceMask:'Soft Difference Mask',normalizeMask:'Normalize Mask Area',directionHint:'+ applies the Target effect, − removes it; both directions alternate automatically.',extraArguments:'Extra CLI Arguments',extraArgumentsPlaceholder:'Example: --seed=42 --max_grad_norm=1.0',startTraining:'Start Training',stopTraining:'Stop',saveConfig:'Save Config',loadConfig:'Load Config',trainingLog:'Training Log',waiting:'Waiting to start…',idle:'Idle',running:'Running',finished:'Finished',failed:'Failed',validating:'Validating and starting…',startFailed:'Failed to start',configLoaded:'Config loaded',invalidConfig:'Invalid config',exitCode:'Exit code',switchLanguage:'中文'},
zh:{subtitle:'Source → Target 配对图片差分 LoRA · 本地端口 3001',modelsDataset:'模型与数据',sourceImages:'Source 图片（编辑前）',targetImages:'Target 图片（编辑后）',pairHint:'两侧图片按文件名（不含扩展名）配对，例如 source/001.png ↔ target/001.png。',outputDirectory:'输出目录',outputName:'输出名称',basicTraining:'基本训练参数',resolution:'分辨率',imageRepeats:'图片重复次数',trainingSteps:'训练步数',saveEvery:'每 N 步保存',learningRate:'学习率',optimizer:'优化器',precision:'精度',sliderScale:'训练 Slider Scale',backgroundWeight:'背景 Loss 权重',minimumTimestep:'最小 Timestep',maximumTimestep:'最大 Timestep',timestepHint:'局部装饰、姿态变化建议 500～1000；颜色和风格变化可从 200～400 开始。',advanced:'高级参数',captionExtension:'Caption 扩展名',captionPrefix:'提示词前缀',captionPrefixPlaceholder:'例如：my_edit_trigger',captionHint:'添加到每张图片 caption 开头；没有同名 caption 文件时也可作为统一提示词。',differenceMask:'软差分 Mask',normalizeMask:'按 Mask 面积归一化',directionHint:'+倍率应用 Target 效果，-倍率移除效果；正反方向自动交替。',extraArguments:'额外命令行参数',extraArgumentsPlaceholder:'例如：--seed=42 --max_grad_norm=1.0',startTraining:'开始训练',stopTraining:'停止训练',saveConfig:'保存配置',loadConfig:'载入配置',trainingLog:'训练日志',waiting:'等待启动…',idle:'空闲',running:'训练中',finished:'已完成',failed:'失败',validating:'正在验证并启动…',startFailed:'启动失败',configLoaded:'配置已载入',invalidConfig:'配置文件无效',exitCode:'退出码',switchLanguage:'English'}
};
let language=localStorage.getItem('animaAddiftLanguage')||((navigator.language||'').toLowerCase().startsWith('zh')?'zh':'en');
const form=document.querySelector('#form'),log=document.querySelector('#log'),start=document.querySelector('#start'),stop=document.querySelector('#stop'),saveConfig=document.querySelector('#saveConfig'),loadConfig=document.querySelector('#loadConfig'),configFile=document.querySelector('#configFile'),langToggle=document.querySelector('#langToggle');
function t(key){return messages[language][key]||key}
function applyLanguage(){
document.documentElement.lang=language==='zh'?'zh-CN':'en';
document.querySelectorAll('[data-i18n]').forEach(e=>e.textContent=t(e.dataset.i18n));
document.querySelectorAll('[data-i18n-placeholder]').forEach(e=>e.placeholder=t(e.dataset.i18nPlaceholder));
langToggle.textContent=t('switchLanguage');
if(first)log.textContent=t('waiting');
updateStatus(lastStatus);
}
langToggle.onclick=()=>{language=language==='en'?'zh':'en';localStorage.setItem('animaAddiftLanguage',language);applyLanguage()};
let last=0, first=true;
let lastStatus={running:false,status:'idle',exit_code:null};
const saved=JSON.parse(localStorage.getItem('animaAddiftConfig')||'{}');
function applyConfig(data){for(const [k,v] of Object.entries(data||{})){const e=form.elements[k];if(e)e.type==='checkbox'?e.checked=!!v:e.value=v}}
applyConfig(saved);
function updateStatus(s){
document.querySelector('#status').textContent=s.running?t('running'):s.status==='finished'?t('finished'):s.status==='failed'?t('failed'):t('idle');
document.querySelector('#dot').className='dot '+(s.running?'running':'');
start.disabled=s.running;stop.disabled=!s.running;
document.querySelector('#meta').textContent=s.running?`PID ${s.pid} · ${s.elapsed}s`:(s.exit_code==null?'':`${t('exitCode')} ${s.exit_code}`);
}
applyLanguage();
function values(){const o={};for(const e of form.elements){if(!e.name)continue;o[e.name]=e.type==='checkbox'?e.checked:e.value}return o}
form.addEventListener('submit',async e=>{e.preventDefault();const data=values();localStorage.setItem('animaAddiftConfig',JSON.stringify(data));start.disabled=true;log.textContent=t('validating')+'\n';last=0;first=false;
try{const r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}),j=await r.json();if(!r.ok)throw Error(j.error||t('startFailed'))}catch(err){log.textContent+='[ERROR] '+err.message+'\n';start.disabled=false}});
stop.onclick=async()=>{stop.disabled=true;await fetch('/api/stop',{method:'POST'})};
saveConfig.onclick=()=>{const data=values();localStorage.setItem('animaAddiftConfig',JSON.stringify(data));const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=(data.output_name||'anima_addift')+'_config.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
loadConfig.onclick=()=>configFile.click();
configFile.onchange=async()=>{const file=configFile.files[0];if(!file)return;try{const data=JSON.parse(await file.text());applyConfig(data);localStorage.setItem('animaAddiftConfig',JSON.stringify(values()));log.textContent=t('configLoaded')+': '+file.name+'\n';first=false}catch(err){log.textContent='[ERROR] '+t('invalidConfig')+': '+err.message+'\n'}finally{configFile.value=''}};
async function poll(){try{const r=await fetch('/api/status?after='+last),s=await r.json();for(const x of s.logs){if(first){log.textContent='';first=false}log.textContent+=`[${x.time}] ${x.text}\n`;last=Math.max(last,x.id);log.scrollTop=log.scrollHeight}
lastStatus=s;updateStatus(s)}catch(e){}setTimeout(poll,1000)}poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "AnimaADDifTWebUI/1.0"

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1024 * 1024:
            raise ValueError("请求内容过大")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/status":
            try:
                query = parsed.query.split("after=", 1)[1] if "after=" in parsed.query else "0"
                after = int(query.split("&", 1)[0])
            except ValueError:
                after = 0
            self._json(STATE.snapshot(after))
            return
        self._json({"error": "Not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/start":
                self._json(start_training(self._body()))
            elif self.path == "/api/stop":
                self._json(stop_training())
            else:
                self._json({"error": "Not found"}, 404)
        except Exception as exc:
            STATE.log(f"ERROR: {exc}", "error")
            self._json({"error": str(exc)}, 400)

    def log_message(self, fmt: str, *args: Any) -> None:
        if "/api/status" not in str(args):
            print(f"[web] {self.address_string()} - {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Anima ADDifT local training WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not TRAIN_SCRIPT.is_file():
        raise SystemExit(f"Training script not found: {TRAIN_SCRIPT}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Anima ADDifT WebUI: {url}")
    print(f"Python: {sys.executable}")
    print(f"Training script: {TRAIN_SCRIPT}")
    if not args.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping WebUI...")
    finally:
        stop_training()
        server.server_close()


if __name__ == "__main__":
    main()
