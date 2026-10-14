"""Explicit-action H2 controls inside the already-running Recovery kernel.

Loading this trusted repository file displays controls only. It neither resets
the runtime nor installs packages, copies images, trains or evaluates anything.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_action(repo, drive_root, action, *, cu_rate=1.54, acknowledgement="", session_hours=0.5,
               detect_python="/content/envs/h1-recovery-detect/bin/python"):
    """A fresh child sees pulled code; stream its output and retain a full log."""
    if action not in {"preflight", "train", "evaluate", "status"}:
        raise ValueError("Only preflight/train/evaluate/status are supported")
    repo, drive_root = Path(repo).resolve(), Path(drive_root).resolve()
    if not math.isfinite(float(cu_rate)) or float(cu_rate) <= 0:
        raise ValueError("CU rate must be positive and finite")
    script = repo / "tools/phase3_h2_roi.py"
    if not script.is_file():
        raise FileNotFoundError("Pull the pushed H2 update in the current notebook first: " + str(script))
    command = [sys.executable, "-u", str(script), action, "--drive-root", str(drive_root), "--cu-rate", str(cu_rate)]
    if action in {"train", "evaluate"}:
        if len(str(acknowledgement)) != 64 or any(c not in "0123456789abcdef" for c in str(acknowledgement)):
            raise RuntimeError("Read and explicitly acknowledge the current preflight/overlays first")
        command += ["--enable", "--acknowledge-preflight", acknowledgement]
        if action == "train":
            if not math.isfinite(float(session_hours)) or not 0 < float(session_hours) <= 2:
                raise ValueError("Session hours must be >0 and <=2")
            command += ["--session-hours", str(session_hours)]
        else:
            command += ["--detect-python", str(detect_python)]
    log_path = drive_root / "phase3/h2_roi_v1_alpha2/logs" / f"continuation_{action}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(command), flush=True)
    print("Full log:", log_path, flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
    if return_code:
        raise RuntimeError(f"H2 {action} stopped with code {return_code}; full log: {log_path}")
    return log_path


def show_h2_controls(repo, drive_root, *, cu_rate=1.54):
    """Linear buttons; the checked hash is held in this kernel, never inferred."""
    try:
        import ipywidgets as widgets
        from IPython.display import Image, Markdown, display
    except ImportError as exc:
        raise RuntimeError("This Colab kernel needs its existing ipywidgets/IPython environment. "
                           "No package was installed and no GPU action launched.") from exc
    repo, drive_root = Path(repo).resolve(), Path(drive_root).resolve()
    preflight_path = drive_root / "phase3/h2_roi_v1_alpha2/preflight.json"
    overlays_dir = preflight_path.parent / "overlays"
    acknowledged = {"sha256": None, "viewed_sha256": None}
    display(Markdown("""### H2 tiếp tục ngay trong phiên này — chỉ một lượt 200 bước

Giữ nguyên L4, ảnh `/content`, môi trường và kết quả H1 đã chạy. Không cần chạy lại
các cell H1 4–8, mở notebook khác hay ngắt/reset runtime.

1. **Chuẩn bị/đọc vùng con vật:** kiểm tra H1 đã hoàn tất, chỉ chép tối đa 10 ảnh
   nếu local còn thiếu và vẽ mask bằng đúng resize/crop/flip của training.
2. Xem các panel, đọc báo cáo rồi **tự tick xác nhận**. Crop ngẫu nhiên không chứa
   con vật vẫn là kết quả thật; không chọn lại seed để làm ảnh trông đẹp hơn.
3. **Train H2 200:** cùng trọng số tác giả và thiết lập H1, optimizer mới; chỉ đổi
   ưu tiên vùng con vật từ 1 lên 2. Khoảng 19 phút phần training, chưa gồm dựng/lưu.
4. Khi đã đủ 200, **chấm H2 trên 30 ảnh**. Dùng lại điểm H1; không sinh lại H1/B0.
   Đánh giá có thể mất khoảng 30 phút hoặc hơn; chỉ chép 30 ảnh còn thiếu.

Không có thao tác nặng tự chạy khi hiển thị bảng. Lỗi thì dùng cell **2A pull**,
chạy lại cell continuation này và chọn đúng thao tác cần tiếp tục. Pull chỉ đổi
file script, không thay các cell đang mở; bảng mới sẽ dùng script vừa pull.
"""))
    preflight = widgets.Button(description="1. Chuẩn bị + xem ROI", layout=widgets.Layout(width="210px"))
    acknowledge = widgets.Checkbox(value=False, description="Tôi đã đọc báo cáo và xem các vùng ROI", indent=False)
    train = widgets.Button(description="2. Train H2 tới 200", button_style="warning", disabled=True,
                           layout=widgets.Layout(width="210px"))
    evaluate = widgets.Button(description="3. Chấm H2 30 ảnh", disabled=True,
                              layout=widgets.Layout(width="210px"))
    status = widgets.Button(description="4. Đọc trạng thái", layout=widgets.Layout(width="210px"))
    session = widgets.BoundedFloatText(value=0.5, min=0.01, max=2, step=0.1,
                                       description="Giờ train:", layout=widgets.Layout(width="210px"))
    output = widgets.Output()
    controls = (preflight, train, evaluate, status)

    def sync_buttons():
        allowed = bool(acknowledged["sha256"] and acknowledge.value)
        train.disabled = not allowed
        evaluate.disabled = not allowed

    def show_evidence():
        if not preflight_path.is_file():
            raise RuntimeError("Chưa có preflight.json. Bấm chuẩn bị trước; không train.")
        report = json.loads(preflight_path.read_text(encoding="utf-8"))
        if report.get("protocol_verified") is not True or report.get("training_allowed") is not True:
            raise RuntimeError("Preflight chưa xác minh/cho phép lượt thử; không mở train.")
        print("Báo cáo:", preflight_path)
        print("SHA256 cần xác nhận:", _sha(preflight_path))
        display(Markdown("```json\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n```"))
        overlay_report = overlays_dir / "report.json"
        if not overlay_report.is_file():
            raise RuntimeError("Thiếu bằng chứng overlay; không mở train.")
        if _sha(overlay_report) != report.get("overlay_report_sha256"):
            raise RuntimeError("Báo cáo overlay đã đổi so với preflight; không mở train.")
        overlays = json.loads(overlay_report.read_text(encoding="utf-8"))
        for case in overlays.get("records", []):
            filename = case.get("panel", "")
            candidate = (overlays_dir / filename).resolve()
            if candidate.parent != overlays_dir.resolve() or not candidate.is_file():
                raise RuntimeError("Panel overlay không hợp lệ; không mở train.")
            if _sha(candidate) != case.get("panel_sha256"):
                raise RuntimeError("Panel overlay đã đổi; chạy lại kiểm tra trước.")
            print(case["image_id"], "ROI crop:", case["roi_fraction"], "ROI latent:", case["latent_roi_fraction"])
            display(Image(filename=str(candidate)))
        print("Nhóm chưa có trong các ảnh đã chọn:", overlays.get("unavailable_cases", []))
        acknowledged["viewed_sha256"] = _sha(preflight_path)

    def ack_changed(change):
        if change["new"]:
            try:
                if not preflight_path.is_file() or not (overlays_dir / "report.json").is_file():
                    raise RuntimeError("Chưa có báo cáo/overlay để xác nhận")
                if acknowledged["viewed_sha256"] != _sha(preflight_path):
                    raise RuntimeError("Bấm chuẩn bị/xem ROI và đọc đúng báo cáo trước khi xác nhận")
                acknowledged["sha256"] = _sha(preflight_path)
                with output:
                    print("Đã xác nhận đúng báo cáo:", acknowledged["sha256"], flush=True)
            except Exception as exc:
                with output:
                    print("Không xác nhận được:", exc, flush=True)
                acknowledge.value = False
                acknowledged["sha256"] = None
        else:
            acknowledged["sha256"] = None
        sync_buttons()

    def perform(action):
        for button in controls:
            button.disabled = True
        acknowledge.disabled = True
        with output:
            try:
                if action == "preflight":
                    acknowledge.value = False
                    acknowledged["sha256"] = None
                    acknowledged["viewed_sha256"] = None
                    run_action(repo, drive_root, action, cu_rate=cu_rate)
                    show_evidence()
                    print("Xem xong rồi tự tick xác nhận; chưa có GPU job nào chạy.", flush=True)
                else:
                    if action in {"train", "evaluate"}:
                        token = acknowledged["sha256"]
                        if not token or not acknowledge.value or _sha(preflight_path) != token:
                            raise RuntimeError("Báo cáo đã đổi hoặc chưa xác nhận. Chuẩn bị/xem ROI rồi tick lại.")
                    run_action(repo, drive_root, action, cu_rate=cu_rate,
                               acknowledgement=acknowledged["sha256"] or "", session_hours=session.value)
            except Exception as exc:
                print("DỪNG an toàn:", exc, flush=True)
            finally:
                preflight.disabled = False
                status.disabled = False
                acknowledge.disabled = False
                sync_buttons()

    acknowledge.observe(ack_changed, names="value")
    for button, action in ((preflight, "preflight"), (train, "train"), (evaluate, "evaluate"), (status, "status")):
        button.on_click(lambda _button, chosen=action: perform(chosen))
    display(widgets.VBox([widgets.HBox([preflight, status]), acknowledge,
                          widgets.HBox([session, train, evaluate]), output]))
    return {"preflight": preflight, "acknowledge": acknowledge, "train": train,
            "evaluate": evaluate, "status": status, "output": output}


if __name__ == "__main__":
    if "REPO" not in globals() or "DRIVE_ROOT" not in globals():
        raise RuntimeError("Run this continuation inside the existing Recovery notebook with REPO and DRIVE_ROOT.")
    H2_CONTROLS = show_h2_controls(REPO, DRIVE_ROOT, cu_rate=globals().get("CU_RATE", 1.54))
