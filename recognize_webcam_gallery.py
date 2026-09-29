import os
import csv
import time
import datetime
import cv2
from picamera2 import Picamera2
import joblib
import numpy as np
from insightface.app import FaceAnalysis
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
GALLERY_PATH = os.path.join(MODEL_DIR, "face_gallery.joblib")


ATTENDANCE_CSV = os.path.join(MODEL_DIR, "attendance.csv")

ATTENDANCE_DETAILS_CSV = os.path.join(MODEL_DIR, "attendance_details.csv")


ENTRY_CONFIRMATION_FRAMES = 3


LEAVE_TIMEOUT = 10

RECOGNITION_THRESHOLD = 0.45

AUDIT_CSV_PATH = os.path.join(MODEL_DIR, "master_recognition_log.csv")
AUDIT_LOG_COOLDOWN_SECONDS = 5.0


REPORT_XLSX_PATH = os.path.join(MODEL_DIR, "attendance_report.xlsx")

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
DET_SIZE = (320, 320)
PROCESS_EVERY_N_FRAMES = 2

print("Loading face gallery...")
gallery = joblib.load(GALLERY_PATH)
print("Known people:", list(gallery.keys()))
for person, vecs in gallery.items():
    print(f"  {person}: {vecs.shape[0]} enrolled photo(s)")
if not gallery:
    print("!! WARNING: gallery is EMPTY — nothing can ever match.")

print("Loading InsightFace (buffalo_l, detection + recognition only)...")
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    allowed_modules=["detection", "recognition"],
)
face_app.prepare(ctx_id=0, det_size=DET_SIZE)
print("Ready.")


def match_identity(embedding: np.ndarray):
   
    query = embedding.astype(np.float32)
    best_person, best_score = None, -1.0
    for person, vecs in gallery.items():
        score = float((vecs @ query).max())
        if score > best_score:
            best_score, best_person = score, person
    return best_person, best_score


def recognize_faces(frame):

    faces = face_app.get(frame)
    results = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        name, score = match_identity(face.normed_embedding)
        results.append((x1, y1, x2, y2, name, score))
    return results


def draw_results(frame, results):
    for (x1, y1, x2, y2, name, score) in results:
        is_match = score >= RECOGNITION_THRESHOLD
        color = (0, 200, 0) if is_match else (0, 0, 200)
        label = f"{name} ({score:.2f})" if is_match else f"Unknown ({score:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
AUDIT_COLUMNS = ["Name", "Confidence (%)", "Time", "Date", "Day", "Bounding Box (x1,y1,x2,y2)", "Frame #"]
_audit_last_logged_at = {}


def init_audit_csv():
    if not os.path.isfile(AUDIT_CSV_PATH):
        with open(AUDIT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(AUDIT_COLUMNS)


def audit_log(name, score, bbox, frame_number, now_dt):
    last = _audit_last_logged_at.get(name)
    if last is not None and (now_dt - last).total_seconds() < AUDIT_LOG_COOLDOWN_SECONDS:
        return
    _audit_last_logged_at[name] = now_dt
    row = [
        name, round(score * 100, 2), now_dt.strftime("%H:%M:%S"), now_dt.strftime("%Y-%m-%d"),
        now_dt.strftime("%A"), f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}", frame_number,
    ]
    with open(AUDIT_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def read_audit_rows():
    if not os.path.isfile(AUDIT_CSV_PATH):
        return []
    with open(AUDIT_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return [[r[0], float(r[1]), r[2], r[3], r[4], r[5], r[6]] for r in reader]


class AttendanceManager:

    def __init__(self, csv_path, details_csv_path, entry_confirmation_frames, leave_timeout):
        self.csv_path = csv_path
        self.details_csv_path = details_csv_path
        self.entry_confirmation_frames = entry_confirmation_frames
        self.leave_timeout = leave_timeout
        self.active_people = {}  # name -> state dict
        self._init_csv()
        self._init_details_csv()
        self._load_open_records()

    def _init_csv(self):
        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["Name", "Date", "Entry Time", "Exit Time"])

    def _init_details_csv(self):
        if not os.path.isfile(self.details_csv_path):
            with open(self.details_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "Name", "Date", "Entry Time",
                    "Entry Accuracy (%)", "Confirmation Frames", "Avg Accuracy While Present (%)",
                ])

    def _load_open_records(self):
        if not os.path.isfile(self.csv_path):
            return
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        now_dt = datetime.datetime.now()
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for name, date_str, entry_str, exit_str in reader:
                if date_str == today_str and exit_str == "":
                    self.active_people[name] = {
                        "status": "present",
                        "confirmation_count": self.entry_confirmation_frames,
                        "confirm_score_sum": 0.0,
                        "first_seen": now_dt,
                        "last_seen": now_dt,
                        "entry_time": now_dt,
                        "csv_row_key": (name, date_str, entry_str),
                        "temporarily_lost": False,
                        "present_score_sum": 0.0,
                        "present_score_count": 0,
                    }
                    print(f"[RESTORED] {name} - resuming session opened at {entry_str}")

    def _record_entry(self, name, dt):
        date_str, entry_str = dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([name, date_str, entry_str, ""])
        return (name, date_str, entry_str)

    def _record_exit(self, row_key, exit_dt):
        name, date_str, entry_str = row_key
        exit_str = exit_dt.strftime("%H:%M:%S")
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header, data_rows = rows[0], rows[1:]
        updated = False
        for r in data_rows:
            if r[0] == name and r[1] == date_str and r[2] == entry_str and r[3] == "":
                r[3] = exit_str
                updated = True
                break
        if not updated:
            print(f"[WARN] Could not find open row for {name} ({date_str} {entry_str}); appending fallback row.")
            data_rows.append([name, date_str, entry_str, exit_str])

        tmp_path = self.csv_path + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows)
        os.replace(tmp_path, self.csv_path)  # atomic swap: a crash mid-write can't corrupt history

    def _record_details_entry(self, row_key, entry_accuracy, confirmation_frames):
        name, date_str, entry_str = row_key
        with open(self.details_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([name, date_str, entry_str, round(entry_accuracy, 2), confirmation_frames, ""])

    def _update_details_avg(self, row_key, avg_accuracy):
        if row_key is None:
            return
        name, date_str, entry_str = row_key
        if not os.path.isfile(self.details_csv_path):
            return
        with open(self.details_csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header, data_rows = rows[0], rows[1:]
        for r in data_rows:
            if r[0] == name and r[1] == date_str and r[2] == entry_str:
                r[5] = round(avg_accuracy, 2)
                break
        tmp_path = self.details_csv_path + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows)
        os.replace(tmp_path, self.details_csv_path)

    def process_recognized_people(self, recognized_scores, now_dt):
        """recognized_scores: dict of name -> similarity score for people
        matched above threshold this processed frame."""
        for name, score in recognized_scores.items():
            person = self.active_people.get(name)

            if person is None:
                self.active_people[name] = {
                    "status": "confirming",
                    "confirmation_count": 1,
                    "confirm_score_sum": score,
                    "first_seen": now_dt,
                    "last_seen": now_dt,
                    "entry_time": None,
                    "csv_row_key": None,
                    "temporarily_lost": False,
                    "present_score_sum": 0.0,
                    "present_score_count": 0,
                }
                print(f"[DETECTED] {name}")
                continue

            person["last_seen"] = now_dt

            if person["status"] == "confirming":
                person["confirmation_count"] += 1
                person["confirm_score_sum"] += score
                print(f"[CONFIRMING] {name} - {person['confirmation_count']}/{self.entry_confirmation_frames}")
                if person["confirmation_count"] >= self.entry_confirmation_frames:
                    person["status"] = "present"
                    person["entry_time"] = now_dt
                    entry_accuracy = (person["confirm_score_sum"] / person["confirmation_count"]) * 100
                    row_key = self._record_entry(name, now_dt)
                    self._record_details_entry(row_key, entry_accuracy, person["confirmation_count"])
                    person["csv_row_key"] = row_key
                    person["present_score_sum"] = score
                    person["present_score_count"] = 1
                    print(f"[ENTRY] {name} - {now_dt.strftime('%H:%M:%S')} (accuracy {entry_accuracy:.1f}%)")
                    print(f"[ACTIVE] {name}")

            elif person["status"] == "present":
                person["present_score_sum"] += score
                person["present_score_count"] += 1
                if person["temporarily_lost"]:
                    person["temporarily_lost"] = False
                    print(f"[ACTIVE] {name}")

    def check_for_exits(self, recognized_names, now_dt):
        for name in list(self.active_people.keys()):
            if name in recognized_names:
                continue  # just handled in process_recognized_people this frame

            person = self.active_people[name]
            elapsed = (now_dt - person["last_seen"]).total_seconds()

            if person["status"] == "confirming":
                if elapsed >= self.leave_timeout:
                    print(f"[TIMEOUT] {name} - never confirmed entry, discarding")
                    del self.active_people[name]

            elif person["status"] == "present":
                if not person["temporarily_lost"]:
                    person["temporarily_lost"] = True
                    print(f"[TEMPORARILY LOST] {name}")
                if elapsed >= self.leave_timeout:
                    self._finalize_exit(name, person, person["last_seen"])
                    del self.active_people[name]

    def _finalize_exit(self, name, person, exit_dt):
        self._record_exit(person["csv_row_key"], exit_dt)
        avg_present_accuracy = (
            (person["present_score_sum"] / person["present_score_count"]) * 100
            if person["present_score_count"] > 0 else None
        )
        if avg_present_accuracy is not None:
            self._update_details_avg(person["csv_row_key"], avg_present_accuracy)
        print(f"[EXIT] {name} - {exit_dt.strftime('%H:%M:%S')}")

    def close_all(self):
       
        now_dt = datetime.datetime.now()
        for name in list(self.active_people.keys()):
            person = self.active_people[name]
            if person["status"] == "present":
                self._finalize_exit(name, person, now_dt)
        self.active_people.clear()



HEADER_FONT = Font(bold=True, color="FFFFFF")
HEADER_FILL = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
GREEN = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
AMBER = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
RED = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
STILL_PRESENT_FILL = PatternFill(start_color="BDD7EE", end_color="BDD7EE", fill_type="solid")
GRAY = PatternFill(start_color="E7E6E6", end_color="E7E6E6", fill_type="solid")


def _style_header(ws, n_cols):
    for col_idx in range(1, n_cols + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font, cell.fill = HEADER_FONT, HEADER_FILL
        cell.alignment = Alignment(horizontal="center")
    ws.freeze_panes = "A2"


def _autosize(ws, headers, rows):
    for col_idx, header in enumerate(headers, start=1):
        max_len = max([len(str(header))] + [len(str(r[col_idx - 1])) for r in rows]) if rows else len(str(header))
        ws.column_dimensions[get_column_letter(col_idx)].width = max_len + 4


def _read_attendance_csv():
    if not os.path.isfile(ATTENDANCE_CSV):
        return []
    with open(ATTENDANCE_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return list(reader)


def _read_details_csv():
    """Returns {(name, date, entry_time): {details}}."""
    details = {}
    if not os.path.isfile(ATTENDANCE_DETAILS_CSV):
        return details
    with open(ATTENDANCE_DETAILS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for name, date_str, entry_str, entry_acc, frames, avg_acc in reader:
            details[(name, date_str, entry_str)] = {
                "entry_accuracy": float(entry_acc) if entry_acc != "" else None,
                "frames": frames if frames != "" else None,
                "avg_accuracy": float(avg_acc) if avg_acc != "" else None,
            }
    return details


def _duration_str(entry_str, exit_str):
    fmt = "%H:%M:%S"
    delta = datetime.datetime.strptime(exit_str, fmt) - datetime.datetime.strptime(entry_str, fmt)
    return str(delta)


def _build_attendance_sheet(wb, attendance_rows, details_map):
    ws = wb.active
    ws.title = "Attendance"
    headers = ["Name", "Accuracy (%)", "Entrance Time", "Leaving Time",
               "Confirmed Via", "Date", "Duration", "Avg Accuracy While Present (%)"]
    ws.append(headers)
    _style_header(ws, len(headers))

    display_rows = []
    for name, date_str, entry_str, exit_str in attendance_rows:
        details = details_map.get((name, date_str, entry_str), {})
        entry_accuracy = details.get("entry_accuracy")
        frames = details.get("frames")
        avg_accuracy = details.get("avg_accuracy")

        still_present = exit_str == ""
        leaving_display = "Still Present" if still_present else exit_str
        duration_display = "Ongoing" if still_present else _duration_str(entry_str, exit_str)
        confirmed_via = (
            f"Face Recognition ({frames} consecutive frames)" if frames
            else "Face Recognition (restored after restart)"
        )
        accuracy_display = round(entry_accuracy, 1) if entry_accuracy is not None else "N/A"
        avg_accuracy_display = round(avg_accuracy, 1) if avg_accuracy is not None else "N/A"

        row = [name, accuracy_display, entry_str, leaving_display, confirmed_via,
               date_str, duration_display, avg_accuracy_display]
        display_rows.append(row)
        ws.append(row)

        fill = (
            GREEN if isinstance(entry_accuracy, float) and entry_accuracy >= 80 else
            AMBER if isinstance(entry_accuracy, float) and entry_accuracy >= 65 else
            RED if isinstance(entry_accuracy, float) else GRAY
        )
        for col_idx in range(1, len(headers) + 1):
            ws.cell(row=ws.max_row, column=col_idx).fill = fill
        if still_present:
            ws.cell(row=ws.max_row, column=4).fill = STILL_PRESENT_FILL
            ws.cell(row=ws.max_row, column=7).fill = STILL_PRESENT_FILL

    ws.auto_filter.ref = ws.dimensions
    _autosize(ws, headers, display_rows)
    return display_rows


def _build_attendance_summary_sheet(wb, attendance_rows, details_map):
    ws = wb.create_sheet("Attendance Summary")
    headers = ["Name", "Total Sessions", "Total Time Present", "Avg Entry Accuracy (%)", "Currently Present?"]
    ws.append(headers)
    _style_header(ws, len(headers))

    per_person = {}
    for name, date_str, entry_str, exit_str in attendance_rows:
        details = details_map.get((name, date_str, entry_str), {})
        entry_accuracy = details.get("entry_accuracy")
        entry = per_person.setdefault(name, {"sessions": 0, "total_seconds": 0, "accuracies": [], "open": False})
        entry["sessions"] += 1
        if entry_accuracy is not None:
            entry["accuracies"].append(entry_accuracy)
        if exit_str == "":
            entry["open"] = True
        else:
            fmt = "%H:%M:%S"
            delta = datetime.datetime.strptime(exit_str, fmt) - datetime.datetime.strptime(entry_str, fmt)
            entry["total_seconds"] += delta.total_seconds()

    display_rows = []
    for name, data in sorted(per_person.items()):
        total_time = str(datetime.timedelta(seconds=int(data["total_seconds"])))
        avg_accuracy = round(sum(data["accuracies"]) / len(data["accuracies"]), 1) if data["accuracies"] else "N/A"
        row = [name, data["sessions"], total_time, avg_accuracy, "Yes" if data["open"] else "No"]
        display_rows.append(row)
        ws.append(row)

    ws.auto_filter.ref = ws.dimensions
    _autosize(ws, headers, display_rows)


def _build_raw_detections_sheets(wb):
    rows = read_audit_rows()
    if not rows:
        return

    ws = wb.create_sheet("Raw Detections")
    ws.append(AUDIT_COLUMNS)
    _style_header(ws, len(AUDIT_COLUMNS))
    for row in rows:
        ws.append(row)
        fill = GREEN if row[1] >= 80 else AMBER if row[1] >= 65 else RED
        for col_idx in range(1, len(AUDIT_COLUMNS) + 1):
            ws.cell(row=ws.max_row, column=col_idx).fill = fill
    ws.auto_filter.ref = ws.dimensions
    _autosize(ws, AUDIT_COLUMNS, rows)

    summary_ws = wb.create_sheet("Raw Detections Summary")
    summary_headers = ["Name", "Sightings", "First Seen", "Last Seen", "Avg Confidence (%)"]
    summary_ws.append(summary_headers)
    _style_header(summary_ws, len(summary_headers))

    per_person = {}
    for name, confidence, t, d, *_ in rows:
        timestamp = f"{d} {t}"
        entry = per_person.setdefault(name, {"count": 0, "first": timestamp, "last": timestamp, "scores": []})
        entry["count"] += 1
        entry["scores"].append(confidence)
        entry["first"] = min(entry["first"], timestamp)
        entry["last"] = max(entry["last"], timestamp)

    summary_rows = []
    for name, entry in sorted(per_person.items()):
        row = [name, entry["count"], entry["first"], entry["last"],
               round(sum(entry["scores"]) / len(entry["scores"]), 2)]
        summary_rows.append(row)
        summary_ws.append(row)
    summary_ws.auto_filter.ref = summary_ws.dimensions
    _autosize(summary_ws, summary_headers, summary_rows)


def export_report():
    attendance_rows = _read_attendance_csv()
    details_map = _read_details_csv()

    if not attendance_rows:
        print("No attendance sessions on record — no Excel report written.")
        return

    wb = Workbook()
    _build_attendance_sheet(wb, attendance_rows, details_map)
    _build_attendance_summary_sheet(wb, attendance_rows, details_map)
    _build_raw_detections_sheets(wb)

    wb.save(REPORT_XLSX_PATH)
    print(f"Attendance report rebuilt: {REPORT_XLSX_PATH}")

init_audit_csv()
attendance = AttendanceManager(ATTENDANCE_CSV, ATTENDANCE_DETAILS_CSV, ENTRY_CONFIRMATION_FRAMES, LEAVE_TIMEOUT)

# ------------------------------------------------------------
# Raspberry Pi Camera (Picamera2)
# ------------------------------------------------------------
print("Starting Raspberry Pi Camera with Picamera2. Press 'q' to quit.")

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (CAPTURE_WIDTH, CAPTURE_HEIGHT),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()

# Give the camera a moment to initialize.
time.sleep(0.3)

frame_count = 0
last_results = []
fps_smoothed = 0.0
prev_time = time.time()

try:
    while True:
        # Picamera2 returns RGB888; OpenCV expects BGR.
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_count += 1

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            last_results = recognize_faces(frame)
            now_dt = datetime.datetime.now()

            recognized_scores = {}
            for (x1, y1, x2, y2, name, score) in last_results:
                if name is not None and score >= RECOGNITION_THRESHOLD:
                    recognized_scores[name] = max(recognized_scores.get(name, 0.0), score)
                    audit_log(name, score, (x1, y1, x2, y2), frame_count, now_dt)

            attendance.process_recognized_people(recognized_scores, now_dt)
            attendance.check_for_exits(set(recognized_scores.keys()), now_dt)

        draw_results(frame, last_results)

        cv2.putText(frame, f"Faces detected: {len(last_results)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        now = time.time()
        instant_fps = 1.0 / max(now - prev_time, 1e-6)
        fps_smoothed = fps_smoothed * 0.9 + instant_fps * 0.1
        prev_time = now
        cv2.putText(frame, f"FPS: {fps_smoothed:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Attendance System (press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    attendance.close_all()
    export_report()