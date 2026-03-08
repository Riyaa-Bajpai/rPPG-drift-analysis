import json
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from datetime import datetime

CV1_PATH = r"D:\project\rppg\cv1_output_20260307_132012.json"


def bandpass(signal, fs):

    low = 0.7
    high = 3.0

    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq

    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)


def compute_hr(signal, fs):

    N = len(signal)

    yf = np.abs(rfft(signal))
    xf = rfftfreq(N, 1/fs)

    mask = (xf >= 0.7) & (xf <= 3)

    xf = xf[mask]
    yf = yf[mask]

    if len(yf) == 0:
        return None

    peak = xf[np.argmax(yf)]

    bpm = peak * 60

    return bpm


def run_cv2():

    with open(CV1_PATH, "r") as f:
        data = json.load(f)

    subjects = data["subjects"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "module": "CV-2",
        "input_from": "CV-1",
        "run_timestamp": timestamp,
        "subjects": {}
    }

    for subject_id, subject in subjects.items():

        print("\nProcessing:", subject_id)

        source_type = subject.get("source_type","")

        if "video" not in source_type:
            print("Skipping image subject")
            continue

        forehead = subject["per_frame_rgb"]["forehead"]

        time = np.array(forehead["time_seconds"])

        R = np.array(forehead["R_per_frame"])
        G = np.array(forehead["G_per_frame"])
        B = np.array(forehead["B_per_frame"])

        fps = 1 / np.mean(np.diff(time))

        rppg = G - (0.5*R + 0.5*B)

        rppg = rppg - np.mean(rppg)

        filtered = bandpass(rppg, fps)

        hr = compute_hr(filtered, fps)

        if hr is None:
            print("HR not detected")
            continue

        print("Real Heart Rate:", round(hr,2), "BPM")

        output["subjects"][subject_id] = {
            "hr_bpm": float(round(hr,2)),
            "method_used": "GREEN",
            "source_type": source_type
        }

    output_path = f"D:/project/rppg/cv2_output_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print("\nCV2 Completed ✅")
    print("Output saved at:", output_path)


if __name__ == "__main__":
    run_cv2()