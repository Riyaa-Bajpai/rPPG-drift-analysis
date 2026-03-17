
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
import json
import os


OUTPUT_DIR    = "cv1_outputs"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
IMAGE_EXTS    = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
VIDEO_EXTS    = ('.mp4', '.avi', '.mov', '.mkv', '.webm')




def safe_save(save_func, primary_path, *args, **kwargs):
    try:
        save_func(primary_path, *args, **kwargs)
        return primary_path
    except OSError:
        base, ext = os.path.splitext(primary_path)
        fallback  = f"{base}_fallback_{RUN_TIMESTAMP}{ext}"
        print(f"  [WARN] Cannot save {os.path.basename(primary_path)} (file open elsewhere?)")
        print(f"         Saving as: {os.path.basename(fallback)}")
        save_func(fallback, *args, **kwargs)
        return fallback



#   FACE DETECTION


def detect_face(image):
    """Detect the largest frontal face. Returns [x,y,w,h] or None."""
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    xml   = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    fc    = cv2.CascadeClassifier(xml)
    faces = fc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        faces = fc.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
    if len(faces) == 0:
        return None
    return faces[np.argmax([w * h for (x, y, w, h) in faces])]

#   ROI DEFINITION


def define_rois(face_rect):
    x, y, w, h = face_rect
    return {
        "forehead":    {"rect": (int(x+w*0.25), int(y+h*0.10), int(w*0.50), int(h*0.18)), "color_bgr": (0, 215, 255),  "label": "Forehead"},
        "left_cheek":  {"rect": (int(x+w*0.05), int(y+h*0.47), int(w*0.28), int(h*0.24)), "color_bgr": (255, 140, 40), "label": "L.Cheek"},
        "right_cheek": {"rect": (int(x+w*0.67), int(y+h*0.47), int(w*0.28), int(h*0.24)), "color_bgr": (60, 210, 60),  "label": "R.Cheek"},
    }


#   RGB EXTRACTION


def extract_roi_rgb(image, rect):
    rx, ry, rw, rh = rect
    H, W = image.shape[:2]
    rx, ry = max(0, rx), max(0, ry)
    rw, rh = min(rw, W - rx), min(rh, H - ry)
    patch  = image[ry:ry+rh, rx:rx+rw]
    if patch.size == 0:
        return {"R": 0.0, "G": 0.0, "B": 0.0}
    m = patch.mean(axis=(0, 1))
    return {"R": round(float(m[2]), 4),
            "G": round(float(m[1]), 4),
            "B": round(float(m[0]), 4)}


#   DRAW ANNOTATIONS ON A FRAME


def draw_annotations(image, face_rect, rois, frame_idx=None, t_sec=None, source=""):
    vis = image.copy()
    x, y, w, h = face_rect

    # face bounding box
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 80), 2)
    cv2.putText(vis, "Face", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)

    # semi-transparent ROI fills
    overlay = vis.copy()
    for roi in rois.values():
        rx, ry2, rw, rh = roi["rect"]
        cv2.rectangle(overlay, (rx, ry2), (rx+rw, ry2+rh), roi["color_bgr"], -1)
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

    # ROI borders + labels
    for roi in rois.values():
        rx, ry2, rw, rh = roi["rect"]
        cv2.rectangle(vis, (rx, ry2), (rx+rw, ry2+rh), roi["color_bgr"], 2)
        cv2.putText(vis, roi["label"], (rx+2, ry2-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, roi["color_bgr"], 1)

    # bottom status bar
    H_img = vis.shape[0]
    if frame_idx is not None and t_sec is not None:
        label = f"[{source}] Frame {frame_idx} | {t_sec:.2f}s"
        cv2.putText(vis, label, (10, H_img - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return vis


#  PROCESS A STATIC IMAGE


def process_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None

    ih, iw    = image.shape[:2]
    face_rect = detect_face(image)
    if face_rect is None:
        mx = int(iw * 0.18); my = int(ih * 0.05)
        face_rect = np.array([mx, my, iw - 2*mx, int(ih * 0.85)])
        print("  WARNING: No face detected, using fallback region")

    rois      = define_rois(face_rect)
    rgb_vals  = {name: extract_roi_rgb(image, roi["rect"]) for name, roi in rois.items()}
    annotated = draw_annotations(image, face_rect, rois, source="IMAGE")

    # simulate 300-frame time-series from mean values
    fps = 30; n = 300; hr = 72
    seeds = {"forehead": 42, "left_cheek": 7, "right_cheek": 99}
    ts_data = {}
    for name, rgb in rgb_vals.items():
        np.random.seed(seeds[name])
        t     = np.linspace(0, n / fps, n)
        f_hr  = hr / 60.0
        pulse = 2.8 * np.sin(2 * np.pi * f_hr * t) + 0.6 * np.sin(4 * np.pi * f_hr * t + 0.3)
        r = rgb["R"] + pulse * 0.50 + np.random.normal(0, 0.9, n)
        g = rgb["G"] + pulse * 1.00 + np.random.normal(0, 0.6, n)
        b = rgb["B"] + pulse * 0.25 + np.random.normal(0, 1.1, n)
        ts_data[name] = {"t": t, "r": r, "g": g, "b": b}

    return {
        "type"      : "image",
        "image"     : image,
        "annotated" : annotated,
        "face_rect" : face_rect,
        "rois"      : rois,
        "rgb_values": rgb_vals,
        "ts"        : ts_data,
        "size"      : (iw, ih),
        "fps"       : fps,
        "n_frames"  : n,
    }


#  PROCESS A PRE-RECORDED VIDEO


def process_video(vid_path, subj, output_dir):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {vid_path}")
        return None

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    iw       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ih       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0

    print(f"  Video   : {iw}x{ih}px | {total} frames | {fps:.1f} fps | {duration:.1f}s")

    ann_vid_path = os.path.join(output_dir, f"cv1_{subj}_annotated_{RUN_TIMESTAMP}.avi")
    fourcc       = cv2.VideoWriter_fourcc(*'XVID')
    writer       = cv2.VideoWriter(ann_vid_path, fourcc, fps, (iw, ih))

    ts_raw = {roi: {"t": [], "r": [], "g": [], "b": []}
              for roi in ["forehead", "left_cheek", "right_cheek"]}

    face_rect = None
    rois      = None
    first_ann = None
    frame_idx = 0
    skipped   = 0

    print(f"  Extracting frames ", end="", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_sec    = frame_idx / fps
        detected = detect_face(frame)

        if detected is not None:
            face_rect = detected
            rois      = define_rois(face_rect)
        elif face_rect is None:
            writer.write(frame)
            frame_idx += 1
            skipped   += 1
            continue

        for name, roi in rois.items():
            rgb = extract_roi_rgb(frame, roi["rect"])
            ts_raw[name]["t"].append(round(t_sec, 4))
            ts_raw[name]["r"].append(rgb["R"])
            ts_raw[name]["g"].append(rgb["G"])
            ts_raw[name]["b"].append(rgb["B"])

        ann_frame = draw_annotations(frame, face_rect, rois, frame_idx, t_sec, "VIDEO")
        writer.write(ann_frame)

        if first_ann is None:
            first_ann = ann_frame.copy()

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(".", end="", flush=True)

    cap.release()
    writer.release()
    print(f" done  ({frame_idx} frames, {skipped} skipped)")

    if first_ann is None:
        print("  [ERROR] No face detected in any frame.")
        return None

    rgb_values = {}
    for name in ts_raw:
        r_arr = np.array(ts_raw[name]["r"])
        g_arr = np.array(ts_raw[name]["g"])
        b_arr = np.array(ts_raw[name]["b"])
        rgb_values[name] = {
            "R": round(float(r_arr.mean()), 2),
            "G": round(float(g_arr.mean()), 2),
            "B": round(float(b_arr.mean()), 2),
        }

    ts_np = {name: {k: np.array(v) for k, v in d.items()} for name, d in ts_raw.items()}

    print(f"  Saved   : {ann_vid_path}")

    return {
        "type"      : "video",
        "image"     : first_ann,
        "annotated" : first_ann,
        "ann_vid"   : ann_vid_path,
        "face_rect" : face_rect,
        "rois"      : rois,
        "rgb_values": rgb_values,
        "ts"        : ts_np,
        "ts_raw"    : ts_raw,
        "size"      : (iw, ih),
        "fps"       : fps,
        "n_frames"  : frame_idx - skipped,
        "duration_s": round(duration, 2),
    }



#   LIVE WEBCAM RECORDING + EXTRACTION


def process_webcam(subj, output_dir, duration_sec=30, camera_index=0):
    """
    Opens webcam live preview window.
    Shows real-time ROI boxes and RGB values while recording.
    Press Q to stop early, or it stops automatically after duration_sec.
    Saves the recorded video + extracts per-frame RGB.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open webcam (index {camera_index})")
        print(f"          Try a different camera index (0, 1, 2...)")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30   # default if webcam reports bad fps
    iw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ih  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Webcam  : {iw}x{ih}px | {fps:.0f} fps")
    print(f"  Recording for {duration_sec}s — press Q in the preview window to stop early")
    print(f"  *** A live preview window will open NOW ***")

    # output recorded video
    webcam_vid_path = os.path.join(output_dir, f"cv1_{subj}_webcam_{RUN_TIMESTAMP}.avi")
    fourcc          = cv2.VideoWriter_fourcc(*'XVID')
    writer          = cv2.VideoWriter(webcam_vid_path, fourcc, fps, (iw, ih))

    ts_raw = {roi: {"t": [], "r": [], "g": [], "b": []}
              for roi in ["forehead", "left_cheek", "right_cheek"]}

    face_rect  = None
    rois       = None
    first_ann  = None
    frame_idx  = 0
    skipped    = 0
    max_frames = int(fps * duration_sec)

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("  [WARN] Webcam frame read failed.")
            break

        t_sec    = frame_idx / fps
        detected = detect_face(frame)

        if detected is not None:
            face_rect = detected
            rois      = define_rois(face_rect)
        elif face_rect is None:
            # no face yet — show plain frame with waiting message
            disp = frame.copy()
            cv2.putText(disp, "Searching for face...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.imshow("CV-1 WEBCAM  |  Press Q to stop", disp)
            writer.write(frame)
            frame_idx += 1
            skipped   += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # extract REAL RGB
        for name, roi in rois.items():
            rgb = extract_roi_rgb(frame, roi["rect"])
            ts_raw[name]["t"].append(round(t_sec, 4))
            ts_raw[name]["r"].append(rgb["R"])
            ts_raw[name]["g"].append(rgb["G"])
            ts_raw[name]["b"].append(rgb["B"])

        ann_frame = draw_annotations(frame, face_rect, rois, frame_idx, t_sec, "WEBCAM")

        # live RGB readout overlay on preview
        fh_rgb = ts_raw["forehead"]
        if fh_rgb["r"]:
            r_val = fh_rgb["r"][-1]; g_val = fh_rgb["g"][-1]; b_val = fh_rgb["b"][-1]
            remaining = max(0, duration_sec - t_sec)
            cv2.putText(ann_frame, f"Forehead  R:{r_val:.0f} G:{g_val:.0f} B:{b_val:.0f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 1)
            cv2.putText(ann_frame, f"Recording... {remaining:.0f}s left  |  Press Q to stop",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

        # progress bar at bottom
        progress = int((frame_idx / max_frames) * iw)
        cv2.rectangle(ann_frame, (0, ih-8), (progress, ih), (0, 200, 100), -1)

        writer.write(ann_frame)
        cv2.imshow("CV-1 WEBCAM  |  Press Q to stop", ann_frame)

        if first_ann is None:
            first_ann = ann_frame.copy()

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n  Stopped early by user at {t_sec:.1f}s")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    actual_frames = frame_idx - skipped
    actual_dur    = frame_idx / fps
    print(f"  Recording done: {actual_frames} frames captured ({actual_dur:.1f}s)")

    if first_ann is None or actual_frames == 0:
        print("  [ERROR] No face detected during webcam session.")
        return None

    # mean RGB
    rgb_values = {}
    for name in ts_raw:
        r_arr = np.array(ts_raw[name]["r"])
        g_arr = np.array(ts_raw[name]["g"])
        b_arr = np.array(ts_raw[name]["b"])
        if len(r_arr) == 0:
            rgb_values[name] = {"R": 0.0, "G": 0.0, "B": 0.0}
        else:
            rgb_values[name] = {
                "R": round(float(r_arr.mean()), 2),
                "G": round(float(g_arr.mean()), 2),
                "B": round(float(b_arr.mean()), 2),
            }

    ts_np = {name: {k: np.array(v) for k, v in d.items()} for name, d in ts_raw.items()}

    print(f"  Saved   : {webcam_vid_path}")

    return {
        "type"      : "webcam",
        "image"     : first_ann,
        "annotated" : first_ann,
        "ann_vid"   : webcam_vid_path,
        "face_rect" : face_rect,
        "rois"      : rois,
        "rgb_values": rgb_values,
        "ts"        : ts_np,
        "ts_raw"    : ts_raw,
        "size"      : (iw, ih),
        "fps"       : fps,
        "n_frames"  : actual_frames,
        "duration_s": round(actual_dur, 2),
    }



#   SUMMARY FIGURE


def save_subject_summary(out_path, annotated, face_rect, rgb_values,
                          ts_data, subj_name, source_type="image"):
    KEYS   = ["forehead", "left_cheek", "right_cheek"]
    COLORS = {"forehead": "#FFD700", "left_cheek": "#4FC3F7", "right_cheek": "#81C784"}
    LABELS = {"forehead": "Forehead", "left_cheek": "Left Cheek", "right_cheek": "Right Cheek"}

    TAG_MAP = {
        "image"  : "IMAGE — Simulated RGB",
        "video"  : "VIDEO — REAL per-frame RGB",
        "webcam" : "WEBCAM — REAL per-frame RGB",
    }
    tag = TAG_MAP.get(source_type, source_type.upper())

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(f"CV-1: Face & ROI Extraction  |  {subj_name}  [{tag}]  |  Drift-Aware rPPG",
                 fontsize=13, fontweight='bold', color='white', y=0.98)

    # annotated image / first frame
    ax_img = fig.add_axes([0.02, 0.10, 0.30, 0.82])
    ax_img.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    ax_img.axis('off')
    ax_img.set_facecolor('#161b22')
    iH, iW = annotated.shape[:2]
    fx, fy, fw, fh = face_rect
    frame_label = {"image": "Image", "video": "First Frame", "webcam": "First Frame"}.get(source_type, "Frame")
    ax_img.set_title(f"{frame_label}: {iW}x{iH}px\nFace  x={fx} y={fy} w={fw} h={fh}",
                     color='white', fontsize=9, pad=6)
    ax_img.legend(handles=[
        Patch(facecolor='#FFD700', label='Forehead ROI'),
        Patch(facecolor='#4FC3F7', label='Left Cheek ROI'),
        Patch(facecolor='#81C784', label='Right Cheek ROI'),
    ], loc='lower center', ncol=3, facecolor='#161b22', labelcolor='white',
       fontsize=8, bbox_to_anchor=(0.5, -0.03))

    # bar chart
    ax_bar = fig.add_axes([0.36, 0.58, 0.24, 0.34])
    ax_bar.set_facecolor('#161b22')
    ax_bar.tick_params(colors='#aaa')
    ax_bar.spines[:].set_color('#333355')
    xp = np.arange(3); wd = 0.25
    ax_bar.bar(xp-wd, [rgb_values[k]["R"] for k in KEYS], wd, label='R', color='#e74c3c', alpha=0.9)
    ax_bar.bar(xp,    [rgb_values[k]["G"] for k in KEYS], wd, label='G', color='#2ecc71', alpha=0.9)
    ax_bar.bar(xp+wd, [rgb_values[k]["B"] for k in KEYS], wd, label='B', color='#3498db', alpha=0.9)
    ax_bar.set_xticks(xp)
    ax_bar.set_xticklabels(["Forehead", "L.Cheek", "R.Cheek"], color='white', fontsize=9)
    ax_bar.set_ylabel("Mean Pixel Intensity", color='#aaa', fontsize=9)
    ax_bar.set_title("Mean RGB per ROI", color='white', fontsize=10)
    ax_bar.legend(facecolor='#0d1117', labelcolor='white', fontsize=8)
    ax_bar.set_ylim(0, 270)

    # data table
    ax_tbl = fig.add_axes([0.36, 0.08, 0.24, 0.42])
    ax_tbl.set_facecolor('#161b22')
    ax_tbl.axis('off')
    ax_tbl.set_title("ROI Values  ->  CV-2", color='white', fontsize=9, pad=6)
    rows = [[LABELS[k], f"{rgb_values[k]['R']:.1f}",
             f"{rgb_values[k]['G']:.1f}", f"{rgb_values[k]['B']:.1f}"] for k in KEYS]
    tbl = ax_tbl.table(cellText=rows, colLabels=["ROI", "R", "G", "B"],
                       loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#0d1117'); cell.set_text_props(color='white')
        cell.set_edgecolor('#333355')
        if r == 0:
            cell.set_facecolor('#21262d')
            cell.set_text_props(color='#58a6ff', fontweight='bold')
    tbl.scale(1, 2.0)

    # time-series
    data_label = "Simulated" if source_type == "image" else "REAL extracted"
    for key, bt in zip(KEYS, [0.67, 0.38, 0.09]):
        ax = fig.add_axes([0.65, bt, 0.33, 0.25])
        ax.set_facecolor('#161b22')
        ax.plot(ts_data[key]['t'], ts_data[key]['g'], color=COLORS[key], lw=1.0,
                label='Green (rPPG)', alpha=0.95)
        ax.plot(ts_data[key]['t'], ts_data[key]['r'], color='#e74c3c', lw=0.7,
                label='Red', alpha=0.6)
        ax.plot(ts_data[key]['t'], ts_data[key]['b'], color='#3498db', lw=0.5,
                label='Blue', alpha=0.45)
        ax.set_title(f"{LABELS[key]}  [{data_label}]", color='white', fontsize=9)
        ax.set_ylabel("Intensity", color='#888', fontsize=7)
        ax.tick_params(colors='#888', labelsize=7)
        ax.spines[:].set_color('#333355')
        ax.grid(True, alpha=0.15, color='#444')
        ax.legend(facecolor='#0d1117', labelcolor='white', fontsize=7, loc='upper right')
        if bt == 0.09:
            ax.set_xlabel("Time (seconds)", color='#aaa', fontsize=8)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()



#  STEP 7 — MULTI-SUBJECT COMPARISON FIGURE


def save_comparison_figure(out_path, all_results):
    KEYS       = ["forehead", "left_cheek", "right_cheek"]
    ROI_COLORS = {"forehead": "#FFD700", "left_cheek": "#4FC3F7", "right_cheek": "#81C784"}
    SUBJ_COLORS= ['#58a6ff', '#ff7f7f', '#7fff7f', '#ffb347', '#da70d6']
    TAG_MAP    = {"image": "[IMAGE]", "video": "[VIDEO]", "webcam": "[WEBCAM]"}
    subjects   = list(all_results.keys())

    fig = plt.figure(figsize=(7 * len(subjects), 13))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle("CV-1: Multi-Subject Comparison  |  Drift-Aware rPPG",
                 fontsize=14, fontweight='bold', color='white', y=0.99)

    col_w = 0.95 / len(subjects)
    for col, subj in enumerate(subjects):
        res  = all_results[subj]
        left = 0.03 + col * col_w
        w    = col_w - 0.03
        tag  = TAG_MAP.get(res["type"], "")

        ax1 = fig.add_axes([left, 0.60, w, 0.35])
        ax1.imshow(cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB))
        ax1.axis('off'); ax1.set_facecolor('#161b22')
        ax1.set_title(f"{subj}  {tag}", color=SUBJ_COLORS[col % len(SUBJ_COLORS)],
                      fontsize=11, fontweight='bold', pad=5)

        ax2 = fig.add_axes([left, 0.33, w, 0.23])
        ax2.set_facecolor('#161b22')
        ax2.tick_params(colors='#aaa'); ax2.spines[:].set_color('#333355')
        rgb = res["rgb_values"]
        xp  = np.arange(3); wd = 0.25
        ax2.bar(xp-wd, [rgb[k]["R"] for k in KEYS], wd, color='#e74c3c', alpha=0.9, label='R')
        ax2.bar(xp,    [rgb[k]["G"] for k in KEYS], wd, color='#2ecc71', alpha=0.9, label='G')
        ax2.bar(xp+wd, [rgb[k]["B"] for k in KEYS], wd, color='#3498db', alpha=0.9, label='B')
        ax2.set_xticks(xp)
        ax2.set_xticklabels(["FH", "LC", "RC"], color='white', fontsize=9)
        ax2.set_ylabel("Mean Intensity", color='#aaa', fontsize=8)
        ax2.set_title("ROI Mean RGB", color='white', fontsize=9)
        ax2.set_ylim(0, 270)
        ax2.legend(facecolor='#0d1117', labelcolor='white', fontsize=7)

        ax3 = fig.add_axes([left, 0.06, w, 0.23])
        ax3.set_facecolor('#161b22')
        for key in KEYS:
            ax3.plot(res["ts"][key]['t'], res["ts"][key]['g'],
                     color=ROI_COLORS[key], lw=1.0,
                     label=key.replace("_", " ").title())
        ax3.set_title("Green Channel — All ROIs", color='white', fontsize=9)
        ax3.set_xlabel("Time (s)", color='#aaa', fontsize=8)
        ax3.set_ylabel("Intensity", color='#888', fontsize=8)
        ax3.tick_params(colors='#888', labelsize=7)
        ax3.spines[:].set_color('#333355')
        ax3.grid(True, alpha=0.15, color='#444')
        ax3.legend(facecolor='#0d1117', labelcolor='white', fontsize=7)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()



#  MAIN FUNCTION

os.makedirs(OUTPUT_DIR, exist_ok=True)
all_results  = {}
input_paths  = []   

print()
print("=" * 62)
print("  CV-1: Face & ROI Extraction  |  Drift-Aware rPPG Project")
print("=" * 62)
print()
print("  Choose input mode for each subject:")
print()
print("    [1] Live Webcam       — records from your webcam now")
print("    [2] Static Image      — passport photo / selfie")
print("    [3] Pre-recorded Video — any face video file")
print()

subject_idx = 1

while True:
    print(f"  ─── Subject_{subject_idx} ───────────────────────────────────")
    mode = input("  Mode (1=Webcam  2=Image  3=Video  ENTER=done): ").strip()

    if mode == "":
        break

    subj = f"Subject_{subject_idx}"
    res  = None

    # WEBCAM 
    if mode == "1":
        print()
        try:
            dur = input("  Record duration in seconds (default 30): ").strip()
            dur = int(dur) if dur.isdigit() else 30
            cam = input("  Camera index (default 0, try 1 if 0 fails): ").strip()
            cam = int(cam) if cam.isdigit() else 0
        except Exception:
            dur = 30; cam = 0

        print(f"  Starting webcam (index={cam}) for {dur}s...")
        res = process_webcam(subj, OUTPUT_DIR, duration_sec=dur, camera_index=cam)
        input_paths.append(f"webcam_index{cam}_{dur}s")

        if res:
            for name, rgb in res["rgb_values"].items():
                print(f"  {name:15s}  R={rgb['R']:6.1f}  G={rgb['G']:6.1f}  B={rgb['B']:6.1f}")

    #  STATIC IMAGE 
    elif mode == "2":
        path = input("  Image path: ").strip()
        if not os.path.isfile(path):
            print(f"  [SKIP] File not found: {path}")
            print()
            continue

        print(f"  Type    : IMAGE")
        res = process_image(path)
        input_paths.append(path)

        if res is None:
            print(f"  [SKIP] Could not read image.")
            continue

        iw, ih = res["size"]
        fx, fy, fw, fh = res["face_rect"]
        print(f"  Loaded  : {iw}x{ih}px")
        print(f"  Face    : x={fx}, y={fy}, w={fw}, h={fh}")
        for name, rgb in res["rgb_values"].items():
            print(f"  {name:15s}  R={rgb['R']:6.1f}  G={rgb['G']:6.1f}  B={rgb['B']:6.1f}")

        ann_path = os.path.join(OUTPUT_DIR, f"cv1_{subj}_annotated_{RUN_TIMESTAMP}.jpg")
        safe_save(cv2.imwrite, ann_path, res["annotated"])

    #  PRE-RECORDED VIDEO 
    elif mode == "3":
        path = input("  Video path: ").strip()
        if not os.path.isfile(path):
            print(f"  [SKIP] File not found: {path}")
            print()
            continue

        ext = os.path.splitext(path)[1].lower()
        if ext not in VIDEO_EXTS:
            print(f"  [SKIP] Unsupported format: {ext}. Use: {VIDEO_EXTS}")
            print()
            continue

        print(f"  Type    : VIDEO")
        res = process_video(path, subj, OUTPUT_DIR)
        input_paths.append(path)

        if res:
            for name, rgb in res["rgb_values"].items():
                print(f"  {name:15s}  R={rgb['R']:6.1f}  G={rgb['G']:6.1f}  B={rgb['B']:6.1f}")

    else:
        print("  Invalid choice. Enter 1, 2, 3 or ENTER to finish.")
        continue

    if res is None:
        print()
        continue

    #  summary figure
    summary_path = os.path.join(OUTPUT_DIR, f"cv1_{subj}_summary_{RUN_TIMESTAMP}.png")
    summary_path = safe_save(save_subject_summary, summary_path,
                             res["annotated"], res["face_rect"],
                             res["rgb_values"], res["ts"], subj, res["type"])
    print(f"  Saved   : {summary_path}")

    all_results[subj] = res
    subject_idx += 1
    print()

if not all_results:
    print("  No subjects processed. Exiting.")
    exit()

#  comparison figure 
if len(all_results) > 1:
    comp_path = os.path.join(OUTPUT_DIR, f"cv1_multi_subject_comparison_{RUN_TIMESTAMP}.png")
    comp_path = safe_save(save_comparison_figure, comp_path, all_results)
    print(f"  Saved   : {comp_path}")

#  JSON output 
payload = {
    "module"       : "CV-1",
    "output_for"   : "CV-2",
    "run_timestamp": RUN_TIMESTAMP,
    "subjects"     : {}
}

for i, (subj, res) in enumerate(all_results.items()):
    fx, fy, fw, fh = res["face_rect"]

    per_frame = {}
    for roi_name in ["forehead", "left_cheek", "right_cheek"]:
        if res["type"] in ("video", "webcam"):
            t_list = res["ts_raw"][roi_name]["t"]
            r_list = res["ts_raw"][roi_name]["r"]
            g_list = res["ts_raw"][roi_name]["g"]
            b_list = res["ts_raw"][roi_name]["b"]
        else:
            ts = res["ts"][roi_name]
            t_list = [round(float(v), 4) for v in ts["t"]]
            r_list = [round(float(v), 4) for v in ts["r"]]
            g_list = [round(float(v), 4) for v in ts["g"]]
            b_list = [round(float(v), 4) for v in ts["b"]]

        per_frame[roi_name] = {
            "time_seconds": t_list,
            "R_per_frame" : r_list,
            "G_per_frame" : g_list,
            "B_per_frame" : b_list,
            "mean_R"      : res["rgb_values"][roi_name]["R"],
            "mean_G"      : res["rgb_values"][roi_name]["G"],
            "mean_B"      : res["rgb_values"][roi_name]["B"],
        }

    src_name = input_paths[i] if i < len(input_paths) else "unknown"
    data_src = {
        "image" : "Simulated from mean RGB",
        "video" : "REAL extracted per-frame",
        "webcam": "REAL extracted per-frame (live webcam)",
    }.get(res["type"], "unknown")

    payload["subjects"][subj] = {
        "source_type"      : res["type"],
        "source_file"      : os.path.basename(src_name),
        "image_size"       : {"width": int(res["size"][0]), "height": int(res["size"][1])},
        "face_bounding_box": {"x": int(fx), "y": int(fy), "w": int(fw), "h": int(fh)},
        "roi_definitions"  : {k: {"rect": list(v["rect"])} for k, v in res["rois"].items()},
        "roi_mean_rgb"     : res["rgb_values"],
        "per_frame_rgb"    : per_frame,
        "timeseries_meta"  : {
            "fps"         : res["fps"],
            "total_frames": res["n_frames"],
            "duration_s"  : round(res.get("duration_s", res["n_frames"] / res["fps"]), 2),
            "data_source" : data_src,
            "note"        : "Green channel = strongest rPPG signal",
        },
    }

def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

json_path = os.path.join(OUTPUT_DIR, f"cv1_output_{RUN_TIMESTAMP}.json")
json_path = safe_save(_write_json, json_path, payload)

#  final summary 
print()
print("=" * 62)
print(f"  CV-1 COMPLETE  ({len(all_results)} subject(s) processed)")
print()
print("  OUTPUT FILES  (in cv1_outputs/ folder):")
for subj, res in all_results.items():
    if res["type"] in ("video", "webcam"):
        print(f"    {os.path.basename(res['ann_vid'])}")
    else:
        print(f"    cv1_{subj}_annotated_{RUN_TIMESTAMP}.jpg")
    print(f"    cv1_{subj}_summary_{RUN_TIMESTAMP}.png")
if len(all_results) > 1:
    print(f"    cv1_multi_subject_comparison_{RUN_TIMESTAMP}.png")
print(f"    {os.path.basename(json_path)}")
print()
print("  JSON data source per subject:")
SRC_LABEL = {
    "image" : "Simulated RGB",
    "video" : "REAL per-frame RGB",
    "webcam": "REAL per-frame RGB (live webcam)",
}
for subj, res in all_results.items():
    print(f"    {subj}: {SRC_LABEL.get(res['type'],'?')}  "
          f"({res['n_frames']} frames @ {res['fps']:.0f}fps)")
print("=" * 62)


