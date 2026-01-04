from __future__ import annotations
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time, date
import uuid
from zoneinfo import ZoneInfo

from PySide6.QtCore import (
    QAbstractTableModel, QModelIndex, Qt, QSortFilterProxyModel, QTimer, Signal, QPoint, QSignalBlocker, QStringListModel
)
from PySide6.QtGui import QAction, QPainter, QPalette
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableView, QMessageBox, QHeaderView,
    QFormLayout, QTextEdit, QDateEdit, QDateTimeEdit, QDialog, QDialogButtonBox,
    QToolBar, QStyleFactory, QSplitter, QTimeEdit, QCompleter
)
from PySide6.QtNetwork import QLocalServer, QLocalSocket

APP_TZ = ZoneInfo("America/New_York")
DB_FILE = "/home/x/work/task_time_tracker.db"
US_LONG_FMT = "%I:%M%p %b %d %Y"   # e.g., 10:00PM Oct 25 2025
US_DATE_FMT = "%m/%d/%Y"           # e.g., 10/25/2025


# ---------- Utilities ----------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def dt_to_iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def iso_to_dt_utc(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def to_local(dt: datetime) -> datetime:
    return dt.astimezone(APP_TZ)


def fmt_local_long(dt: datetime) -> str:
    s = to_local(dt).strftime(US_LONG_FMT)
    return s.lstrip("0")  # remove leading 0 hour


def fmt_local_date(dt: datetime) -> str:
    return to_local(dt).strftime(US_DATE_FMT)


def seconds_to_hms(total: int) -> str:
    h, r = divmod(int(total), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def local_yyyy_mm_dd(dt: datetime) -> str:
    return to_local(dt).strftime("%Y-%m-%d")


# ---------- Data layer ----------

class Database:
    def __init__(self, path: str = DB_FILE):
        self.path = path
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS task_entries (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              note TEXT DEFAULT '',
              start_utc TEXT NOT NULL,
              end_utc   TEXT NOT NULL,
              duration_seconds INTEGER NOT NULL,
              local_date TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_task_entries_local_date ON task_entries(local_date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_task_entries_name ON task_entries(name)")
        self._conn.commit()
        self._ensure_uid_column()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS running_task (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                name TEXT NOT NULL,
                note TEXT DEFAULT '',
                start_utc TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    # CRUD
    def insert_entry(self, name: str, note: str, start: datetime, end: datetime):
        du = int((end - start).total_seconds())
        local_date_str = local_yyyy_mm_dd(start)
        uid = str(uuid.uuid4())
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO task_entries(uid, name, note, start_utc, end_utc, duration_seconds, local_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (uid, name, note or "", dt_to_iso_utc(start), dt_to_iso_utc(end), du, local_date_str),
        )
        self._conn.commit()
        return uid

    def delete_entries(self, ids: list[int]):
        if not ids:
            return
        q = f"DELETE FROM task_entries WHERE id IN ({','.join('?'*len(ids))})"
        self._conn.execute(q, ids)
        self._conn.commit()

    def update_entry(self, id_: int, *, name: str | None = None, note: str | None = None,
                      start: datetime | None = None, end: datetime | None = None):
        # Load existing
        cur = self._conn.cursor()
        row = cur.execute("SELECT * FROM task_entries WHERE id=?", (id_,)).fetchone()
        if not row:
            raise ValueError("Row not found")
        name = name if name is not None else row["name"]
        note = note if note is not None else row["note"]
        start = start if start is not None else iso_to_dt_utc(row["start_utc"])  # utc
        end = end if end is not None else iso_to_dt_utc(row["end_utc"])        # utc
        du = int((end - start).total_seconds())
        local_date_str = local_yyyy_mm_dd(start)
        cur.execute(
            """
            UPDATE task_entries
               SET name=?, note=?, start_utc=?, end_utc=?, duration_seconds=?, local_date=?,
                   updated_at=CURRENT_TIMESTAMP
             WHERE id=?
            """,
            (name, note or "", dt_to_iso_utc(start), dt_to_iso_utc(end), du, local_date_str, id_),
        )
        self._conn.commit()

    def fetch_today(self, today_local: str):
        cur = self._conn.cursor()
        return cur.execute(
            "SELECT * FROM task_entries WHERE local_date=? ORDER BY start_utc DESC",
            (today_local,),
        ).fetchall()

    def fetch_all(self):
        cur = self._conn.cursor()
        return cur.execute("SELECT * FROM task_entries ORDER BY start_utc DESC").fetchall()

    def fetch_today_totals(self, today_local: str):
        """Return rows of (name, total) for today's cumulative seconds per task."""
        cur = self._conn.cursor()
        return cur.execute(
            """
            SELECT name, SUM(duration_seconds) AS total
            FROM task_entries
            WHERE local_date=?
            GROUP BY name
            ORDER BY total DESC, name COLLATE NOCASE ASC
            """,
            (today_local,),
        ).fetchall()

    def search(self, text: str, dfrom: str | None, dto: str | None):
        # dfrom and dto are YYYY-MM-DD local
        params = []
        where = []
        if text:
            like = f"%{text.lower()}%"
            where.append("(lower(name) LIKE ? OR lower(note) LIKE ?)")
            params += [like, like]
        if dfrom:
            where.append("local_date >= ?")
            params.append(dfrom)
        if dto:
            where.append("local_date <= ?")
            params.append(dto)
        where_clause = (" WHERE " + " AND ".join(where)) if where else ""
        q = (
            "SELECT * FROM task_entries" + where_clause + " ORDER BY start_utc DESC"
        )
        cur = self._conn.cursor()
        return cur.execute(q, params).fetchall()

    def suggest_task_name(self, prefix: str) -> str | None:
        """Return the most recently used task name beginning with `prefix` (case-insensitive)."""
        prefix = (prefix or "").strip().lower()
        if not prefix:
            return None
        cur = self._conn.cursor()
        row = cur.execute(
            """
            SELECT name, MAX(start_utc) AS last_use
            FROM task_entries
            WHERE lower(name) LIKE ?
            GROUP BY name
            ORDER BY last_use DESC
            LIMIT 1
            """,
            (prefix + "%",),
        ).fetchone()
        return row["name"] if row else None

    def suggest_task_names(self, prefix: str, limit: int = 25) -> list[str]:
        """Return recent distinct task names beginning with `prefix` (case-insensitive)."""
        prefix = (prefix or "").strip().lower()
        if not prefix:
            return []
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT name, MAX(start_utc) AS last_use
            FROM task_entries
            WHERE lower(name) LIKE ?
            GROUP BY lower(name)
            ORDER BY last_use DESC
            LIMIT ?
            """,
            (prefix + "%", limit),
        ).fetchall()
        return [r["name"] for r in rows]

    def _ensure_uid_column(self):
        cur = self._conn.cursor()
        # check schema
        cols = cur.execute("PRAGMA table_info(task_entries)").fetchall()
        has_uid = any(c["name"] == "uid" for c in cols)
        if not has_uid:
            cur.execute("ALTER TABLE task_entries ADD COLUMN uid TEXT")
            self._conn.commit()
        # backfill missing
        cur = self._conn.cursor()
        missing = cur.execute("SELECT id FROM task_entries WHERE uid IS NULL OR uid=''").fetchall()
        for r in missing:
            cur.execute("UPDATE task_entries SET uid=? WHERE id=?", (str(uuid.uuid4()), r["id"]))
        # indexes
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_task_entries_uid ON task_entries(uid)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_task_entries_local_name ON task_entries(local_date, name)")
        self._conn.commit()

    # UUID based CRUD
    def fetch_entry_by_uid(self, uid: str):
        return self._conn.cursor().execute("SELECT * FROM task_entries WHERE uid=?", (uid,)).fetchone()

    def update_entry_by_uid(self, uid: str, *, name: str | None = None,
                            note: str | None = None, start: datetime | None = None, end: datetime | None = None):
        cur = self._conn.cursor()
        row = cur.execute("SELECT * FROM task_entries WHERE uid=?", (uid,)).fetchone()
        if not row:
            raise ValueError("Row not found")
        name = name if name is not None else row["name"]
        note = note if note is not None else row["note"]
        start = start if start is not None else iso_to_dt_utc(row["start_utc"])
        end = end if end is not None else iso_to_dt_utc(row["end_utc"])
        du = int((end - start).total_seconds())
        local_date_str = local_yyyy_mm_dd(start)
        cur.execute(
            """
            UPDATE task_entries
               SET name=?, note=?, start_utc=?, end_utc=?, duration_seconds=?, local_date=?,
                   updated_at=CURRENT_TIMESTAMP
             WHERE uid=?
            """,
            (name, note or "", dt_to_iso_utc(start), dt_to_iso_utc(end), du, local_date_str, uid),
        )
        self._conn.commit()

    def delete_entries_by_uids(self, uids: list[str]):
        if not uids:
            return
        q = f"DELETE FROM task_entries WHERE uid IN ({','.join('?'*len(uids))})"
        self._conn.execute(q, uids)
        self._conn.commit()

    def delete_today_by_name(self, today_local: str, name: str):
        self._conn.execute(
            "DELETE FROM task_entries WHERE local_date=? AND name=?",
            (today_local, name)
        )
        self._conn.commit()

    def fetch_recent_task_names(self, limit: int = 500) -> list[tuple[str, str]]:
        """
        Return distinct task names with their most recent use (ISO string), most-recent first.
        """
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT name, MAX(start_utc) AS last_use
            FROM task_entries
            GROUP BY lower(name)
            ORDER BY last_use DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [(r["name"], r["last_use"]) for r in rows]

    # save/rescue running tasks

    def set_running_task(self, name: str, note: str, start: datetime):
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO running_task (id, name, note, start_utc)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                note=excluded.note,
                start_utc=excluded.start_utc
            """,
            (name, note or "", dt_to_iso_utc(start)),
        )
        self._conn.commit()

    def clear_running_task(self):
        self._conn.execute("DELETE FROM running_task WHERE id=1")
        self._conn.commit()

    def get_running_task(self):
        cur = self._conn.cursor()
        return cur.execute("SELECT name, note, start_utc FROM running_task WHERE id=1").fetchone()


class FuzzyCompletingLineEdit(QLineEdit):
    """
    Fuzzy search + live dropdown using QCompleter:
      • No auto-mutating text while you type (deletes feel natural).
      • Dropdown shows ranked matches; click/Enter/Tab accepts.
      • Best match cached for external use (e.g., Start button).
    """
    def __init__(self, db: Database, *args, max_results: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self._db = db
        self._max_results = max_results
        self._all_names: list[str] = []
        self._recency_rank: dict[str, int] = {}
        self._best: str | None = None

        # Model + completer
        self._model = QStringListModel(self)
        self._completer = QCompleter(self._model, self)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.setCompleter(self._completer)

        self._completer.activated[str].connect(self._on_completer_activated)
        self.textEdited.connect(self._on_text_edited)

        self.setPlaceholderText("Type task… (fuzzy search)")
        self.refresh_names()

    # ---- Public API ----
    def refresh_names(self):
        recent = self._db.fetch_recent_task_names(limit=800)
        self._all_names = [n for (n, _) in recent]
        self._recency_rank = {n.lower(): i for i, (n, _) in enumerate(recent)}
        self._refresh_for_current_text()

    def best_match(self) -> str | None:
        return self._best

    # ---- Events ----
    def focusInEvent(self, e):
        super().focusInEvent(e)
        # Show recent items on focus if empty
        self._refresh_for_current_text(show_even_if_empty=True)

    def keyPressEvent(self, e):
        # If popup isn't visible, Enter/Tab should accept the best match
        if e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Tab) and not self._completer.popup().isVisible():
            if self._best:
                self.setText(self._best)
                self.setCursorPosition(len(self._best))
                # fall through so editingFinished etc. still fire
        super().keyPressEvent(e)

    # ---- Slots ----
    def _on_text_edited(self, _):
        self._refresh_for_current_text()

    def _on_completer_activated(self, text: str):
        self.setText(text)
        self.setCursorPosition(len(text))

    # ---- Core ----
    def _refresh_for_current_text(self, show_even_if_empty: bool = False):
        q = self.text().strip()
        if q or show_even_if_empty:
            matches = self._ranked_matches(q) if q else self._ranked_matches("")
            self._best = matches[0] if matches else None
            self._model.setStringList(matches[: self._max_results])
            if matches:
                # Positions and shows the dropdown under the line edit
                self._completer.complete()
            else:
                self._completer.popup().hide()
        else:
            self._best = None
            self._model.setStringList([])
            self._completer.popup().hide()

    # ---- Fuzzy ranking (same idea as before, but we never mutate text) ----
    def _ranked_matches(self, query: str) -> list[str]:
        q = query.lower()
        scored = []
        for name in self._all_names:
            nl = name.lower()
            prefix = nl.startswith(q) if q else False
            subpos = nl.find(q) if q else 0
            subseq_gap = self._subsequence_gap(q, nl) if q else 0
            if q and (subpos < 0) and (subseq_gap is None):
                continue
            length = len(name)
            rec = self._recency_rank.get(nl, 10_000)
            key = (
                0 if prefix else 1,
                subpos if subpos >= 0 else 9999,
                subseq_gap if subseq_gap is not None else 9999,
                length,
                rec,
            )
            scored.append((key, name))
        scored.sort(key=lambda t: t[0])
        return [n for _, n in scored]

    @staticmethod
    def _subsequence_gap(q: str, s: str) -> int | None:
        if not q:
            return 0
        i = 0
        last = -1
        gap_sum = 0
        for ch in q:
            i = s.find(ch, i)
            if i == -1:
                return None
            if last != -1:
                gap_sum += (i - last - 1)
            last = i
            i += 1
        return gap_sum


# ---------- Table models ----------

class EntriesTableModel(QAbstractTableModel):
    headers = ["ID", "Task", "Start", "End", "Duration", "Date", "Note"]

    def __init__(self, rows: list[sqlite3.Row] | None = None):
        super().__init__()
        self._rows: list[sqlite3.Row] = rows or []

    def set_rows(self, rows: list[sqlite3.Row]):
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    # Qt model API
    def rowCount(self, parent=QModelIndex()):
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        col = index.column()
        if role in (Qt.DisplayRole, Qt.EditRole):
            if col == 0:
                return row["id"]
            elif col == 1:
                return row["name"]
            elif col == 2:
                return fmt_local_long(iso_to_dt_utc(row["start_utc"]))
            elif col == 3:
                return fmt_local_long(iso_to_dt_utc(row["end_utc"]))
            elif col == 4:
                return seconds_to_hms(row["duration_seconds"])
            elif col == 5:
                # Display local date as US mm/dd/yyyy
                dt = iso_to_dt_utc(row["start_utc"])  # base on start
                return fmt_local_date(dt)
            elif col == 6:
                return row["note"]
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.headers[section]
        return section + 1

    def row(self, r: int) -> sqlite3.Row:
        return self._rows[r]


class TotalsTableModel(QAbstractTableModel):
    headers = ["Task", "Total", "Actions"]

    def __init__(self, rows=None):
        super().__init__()
        self._rows = rows or []  # rows are sqlite3.Row or (name, total_seconds)

    def set_rows(self, rows):
        self.beginResetModel()
        self._rows = rows or []
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 3

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        r = self._rows[index.row()]
        # support sqlite3.Row or (name, total) tuple
        name = r["name"] if isinstance(r, sqlite3.Row) else r[0]
        total = r["total"] if isinstance(r, sqlite3.Row) else r[1]
        if index.column() == 0:
            return name
        if index.column() == 1:
            return seconds_to_hms(int(total))
        if index.column() == 2:  # actions
            return ""
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.headers[section]
        return section + 1

    def sort(self, column, order):
        if column == 2:  # ignore actions
            return
        # Sort by name (0) or total seconds (1)
        rev = (order == Qt.DescendingOrder)

        def key(r):
            if column == 0:
                return (r["name"] if isinstance(r, sqlite3.Row) else r[0]).lower()
            else:
                return int(r["total"] if isinstance(r, sqlite3.Row) else r[1])

        self.layoutAboutToBeChanged.emit()
        self._rows.sort(key=key, reverse=rev)
        self.layoutChanged.emit()


class FilterProxy(QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self.search_text = ""

    @property
    def current_search(self) -> str:
        return getattr(self, "_current_search", "")

    def beginFilterChange(self):
        # QSortFilterProxyModel doesn't expose begin/end for filter changes publicly,
        # but we keep a flag to avoid excessive invalidations.
        self._current_search = self.search_text

    def endFilterChange(self, _direction):
        # Trigger a refilter
        self.invalidateFilter()

    def set_search_text(self, text: str):
        new_text = (text or "").lower()
        if new_text == self.current_search:
            return
        self.beginFilterChange()
        self.search_text = new_text
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(self, source_row, source_parent):
        if not self.search_text:
            return True
        m: EntriesTableModel = self.sourceModel()  # type: ignore
        for c in range(m.columnCount()):
            idx = m.index(source_row, c)
            v = m.data(idx, Qt.DisplayRole)
            if v is None:
                continue
            if self.search_text in str(v).lower():
                return True
        return False


# ---------- Edit dialog ----------

class EditDialog(QDialog):
    saved = Signal()

    def __init__(self, db: Database, row: sqlite3.Row, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Entry")
        self.db = db
        self.row = row

        form = QFormLayout()
        self.name_edit = QLineEdit(row["name"])
        self.note_edit = QTextEdit()
        self.note_edit.setPlainText(row["note"] or "")

        sdt = to_local(iso_to_dt_utc(row["start_utc"]))
        edt = to_local(iso_to_dt_utc(row["end_utc"]))

        self.start_date = QDateEdit(sdt.date())
        self.start_date.setDisplayFormat("MM/dd/yyyy")
        self.start_date.setCalendarPopup(True)

        self.start_time = QTimeEdit(sdt.time())
        self.start_time.setDisplayFormat("h:mm AP")

        self.end_date = QDateEdit(edt.date())
        self.end_date.setDisplayFormat("MM/dd/yyyy")
        self.end_date.setCalendarPopup(True)

        self.end_time = QTimeEdit(edt.time())
        self.end_time.setDisplayFormat("h:mm AP")

        # Row widgets for form layout
        start_row = QWidget()
        start_row_lay = QHBoxLayout(start_row)
        start_row_lay.setContentsMargins(0, 0, 0, 0)
        start_row_lay.addWidget(self.start_date, 1)
        start_row_lay.addWidget(self.start_time, 1)

        end_row = QWidget()
        end_row_lay = QHBoxLayout(end_row)
        end_row_lay.setContentsMargins(0, 0, 0, 0)
        end_row_lay.addWidget(self.end_date, 1)
        end_row_lay.addWidget(self.end_time, 1)

        form.addRow("Task", self.name_edit)
        form.addRow("Note", self.note_edit)
        form.addRow("Start", start_row)
        form.addRow("End", end_row)

        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_save)
        btns.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(btns)

    def _on_save(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid", "Task name is required")
            return
        s_local = datetime.combine(
            self.start_date.date().toPython(),
            self.start_time.time().toPython(),
            tzinfo=APP_TZ
        )
        e_local = datetime.combine(
            self.end_date.date().toPython(),
            self.end_time.time().toPython(),
            tzinfo=APP_TZ
        )

        # sanity checks
        if e_local <= s_local:
            QMessageBox.warning(self, "Invalid", "End must be after start.")
            return
        if s_local.year >= 3000 or e_local.year >= 3000:
            QMessageBox.warning(self, "Invalid", "Year must be before 3000.")
            return
        if local_yyyy_mm_dd(s_local) != local_yyyy_mm_dd(e_local):
            QMessageBox.warning(self, "Invalid", "Start and end must be on the same local day.")
            return

        s_utc = s_local.astimezone(timezone.utc)
        e_utc = e_local.astimezone(timezone.utc)
        try:
            uid = self.row["uid"] if "uid" in self.row.keys() else None
            if uid:
                self.db.update_entry_by_uid(uid, name=name, note=self.note_edit.toPlainText(),
                                            start=s_utc, end=e_utc)
            else:
                # fallback for legacy rows (shouldn’t happen after backfill)
                self.db.update_entry(self.row["id"], name=name, note=self.note_edit.toPlainText(),
                                     start=s_utc, end=e_utc)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        self.saved.emit()
        self.accept()


# ---------- Tabs ----------

class TimerTab(QWidget):
    data_changed = Signal()
    title_changed = Signal(str)

    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self.active_name: str | None = None
        self.active_note: str = ""
        self.active_start: datetime | None = None

        # UI
        main = QVBoxLayout(self)

        # Controls
        row = QHBoxLayout()
        self.name_edit = FuzzyCompletingLineEdit(self.db)

        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("Optional note")
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        row.addWidget(self.name_edit, 3)
        row.addWidget(self.note_edit, 2)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        main.addLayout(row)

        # Active info
        info = QHBoxLayout()
        self.active_label = QLabel("No active task")
        self.elapsed_label = QLabel("00:00:00")
        self.elapsed_label.setStyleSheet("font: 24px; font-weight: 700;")
        info.addWidget(self.active_label)
        info.addStretch()
        info.addWidget(QLabel("Elapsed:"))
        info.addWidget(self.elapsed_label)
        main.addLayout(info)

        # Today section
        self.table_model = EntriesTableModel([])
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        self.table.setColumnHidden(0, True)  # hide ID

        self.totals_model = TotalsTableModel([])
        self.totals_table = QTableView()
        self.totals_table.setModel(self.totals_model)
        self.totals_table.setSelectionBehavior(QTableView.SelectRows)
        self.totals_table.setSelectionMode(QTableView.SingleSelection)
        self.totals_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.totals_table.setAlternatingRowColors(True)
        self.totals_table.setSortingEnabled(True)

        # add widgets
        main.addWidget(self.table, 1)
        main.addWidget(self.totals_table, 1)

        # refresh buttons after sort
        self.totals_model.layoutChanged.connect(self._rebuild_totals_action_widgets)

        # footer/total
        self.total_label = QLabel("Today total: 00:00:00")
        self.total_label.setStyleSheet("color: #666; font-size: 12px; padding: 2px;")
        main.addWidget(self.total_label)

        # Timer
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)

        self.refresh_today()
        self.data_changed.connect(self.name_edit.refresh_names)
        self._last_local_date = local_yyyy_mm_dd(now_utc())

        # watch for day rollover
        self._midnight_watch = QTimer(self)
        self._midnight_watch.setTimerType(Qt.VeryCoarseTimer)
        self._midnight_watch.setInterval(30_000)               # check every 30s
        self._midnight_watch.timeout.connect(self._check_midnight_rollover)
        self._midnight_watch.start()

        # check for running task
        self._restore_running_task()

    def _start(self):
        if self.active_start is not None:
            QMessageBox.information(self, "Already running", "Stop the current task first")
            return

        typed = self.name_edit.text().strip()
        best = self.name_edit.best_match()

        name = typed
        if best:
            tl = typed.lower()
            bl = best.lower()
            if tl and bl.startswith(tl) and bl != tl:
                name = best  # only snap on real prefix

        if not name:
            QMessageBox.warning(self, "Invalid", "Task name is required")
            return

        self.active_name = name
        self.active_note = self.note_edit.text().strip()
        self.active_start = now_utc()

        self.db.set_running_task(self.active_name, self.active_note, self.active_start)

        self.active_label.setText(f"Active: {self.active_name}")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.timer.start()
        self._emit_title(0)

    def _stop(self):
        if self.active_start is None:
            return
        end = now_utc()
        try:
            self.db.insert_entry(self.active_name or "", self.active_note, self.active_start, end)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        # reset
        self.db.clear_running_task()
        self.active_name = None
        self.active_note = ""
        self.active_start = None
        self.active_label.setText("No active task")
        self.elapsed_label.setText("00:00:00")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.timer.stop()
        self.refresh_today()
        self.data_changed.emit()
        self._emit_title(None)

    def _tick(self):
        if self.active_start is None:
            return
        du = int((now_utc() - self.active_start).total_seconds())
        self.elapsed_label.setText(seconds_to_hms(du))
        self._emit_title(du)
        self._check_midnight_rollover()

    def refresh_today(self):
        today_local = local_yyyy_mm_dd(now_utc())

        # Top table (entries)
        rows = self.db.fetch_today(today_local)
        self.table_model.set_rows(rows)
        self.table.setColumnHidden(0, True)  # hide ID
        self._rebuild_entry_action_widgets()

        # Bottom table (per-task totals)
        totals_rows = self.db.fetch_today_totals(today_local)
        self.totals_model.set_rows(totals_rows)
        self._rebuild_totals_action_widgets()

        # Footer: cumulative total for the day
        total = sum(r["duration_seconds"] for r in rows)
        self.total_label.setText(f"Today total: {seconds_to_hms(total)}")

    def _emit_title(self, du: int | None = None):
        """While running: window title is just the ticking timer (HH:MM:SS). Otherwise: app name."""
        if self.active_start is None:
            self.title_changed.emit("Task Time Tracker")
            return
        if du is None:
            du = int((now_utc() - self.active_start).total_seconds())
        self.title_changed.emit(seconds_to_hms(du))

    # --- actions: top table ---
    def _rebuild_entry_action_widgets(self):
        # create Edit/Delete buttons per row in the last column
        model = self.table_model
        last_col = model.columnCount() - 1
        for r in range(model.rowCount()):
            idx = model.index(r, last_col)
            row = model.row(r)
            uid = row["uid"]  # guaranteed after backfill

            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)

            btn_edit = QPushButton("Edit")
            btn_edit.setProperty("uid", uid)
            btn_edit.clicked.connect(lambda _=False, u=uid: self._on_edit_entry(u))

            btn_del = QPushButton("Delete")
            btn_del.setStyleSheet("color:#a00;")
            btn_del.setProperty("uid", uid)
            btn_del.clicked.connect(lambda _=False, u=uid: self._on_delete_entry(u))

            h.addWidget(btn_edit)
            h.addWidget(btn_del)
            h.addStretch(1)
            self.table.setIndexWidget(idx, w)

        # give the actions column a reasonable width
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(last_col, 160)

    def _on_edit_entry(self, uid: str):
        row = self.db.fetch_entry_by_uid(uid)
        if not row:
            QMessageBox.warning(self, "Not found", "That entry no longer exists.")
            return
        dlg = EditDialog(self.db, row, self)
        dlg.saved.connect(self._after_data_change)
        dlg.exec()

    def _on_delete_entry(self, uid: str):
        row = self.db.fetch_entry_by_uid(uid)
        if not row:
            QMessageBox.information(self, "Gone", "That entry is already deleted.")
            return
        name = row["name"]
        start_local = fmt_local_long(iso_to_dt_utc(row["start_utc"]))
        if QMessageBox.question(
            self, "Confirm delete",
            f"Delete this entry?\n\nTask: {name}\nStart: {start_local}"
        ) != QMessageBox.Yes:
            return
        self.db.delete_entries_by_uids([uid])
        self._after_data_change()

    # --- actions: bottom totals table ---
    def _rebuild_totals_action_widgets(self):
        model = self.totals_model
        last_col = model.columnCount() - 1
        for r in range(model.rowCount()):
            idx = model.index(r, last_col)
            # rows from fetch_today_totals are sqlite3.Row or tuples; unify access
            row = model._rows[r]
            name = row["name"] if isinstance(row, sqlite3.Row) else row[0]

            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)

            btn_del = QPushButton("Delete")
            btn_del.setStyleSheet("color:#a00;")
            btn_del.clicked.connect(lambda _=False, nm=name: self._on_delete_all_today_for_task(nm))

            h.addWidget(btn_del)
            h.addStretch(1)
            self.totals_table.setIndexWidget(idx, w)

        self.totals_table.horizontalHeader().setStretchLastSection(False)
        self.totals_table.setColumnWidth(last_col, 100)

    def _on_delete_all_today_for_task(self, name: str):
        today_local = local_yyyy_mm_dd(now_utc())
        if QMessageBox.question(
            self, "Confirm delete all for today",
            f"Delete ALL entries for today named:\n\n{name}\n\nThis cannot be undone."
        ) != QMessageBox.Yes:
            return
        self.db.delete_today_by_name(today_local, name)
        self._after_data_change()

    # --- common refresh hook after edits/deletes/insert ---
    def _after_data_change(self):
        self.refresh_today()
        self.data_changed.emit()

    def _check_midnight_rollover(self):
        """Detect local-day change; if a task is still running, end it at local midnight.
        Then refresh UI to show the new day."""
        now = now_utc()
        now_local_date = local_yyyy_mm_dd(now)

        if now_local_date == self._last_local_date:
            return  # still same local day

        # We just crossed midnight
        try:
            #  If a task is still running, end it exactly at midnight local
            if self.active_start is not None:
                # boundary is 00:00 of the new local date
                boundary_local = datetime.combine(date.fromisoformat(now_local_date), time.min, tzinfo=APP_TZ)
                boundary_utc = boundary_local.astimezone(timezone.utc)

                # Only insert if the active start was before the boundary
                if self.active_start < boundary_utc:
                    self.db.insert_entry(self.active_name or "", self.active_note, self.active_start, boundary_utc)

                # Reset active UI state
                self.active_name = None
                self.active_note = ""
                self.active_start = None
                self.active_label.setText("No active task")
                self.elapsed_label.setText("00:00:00")
                self.btn_start.setEnabled(True)
                self.btn_stop.setEnabled(False)
                self.db.clear_running_task()
                self.timer.stop()

            #  clear/refresh UI for the new day
            self._last_local_date = now_local_date
            self.name_edit.clear()
            self.note_edit.clear()

            self.refresh_today()
            self.data_changed.emit()
            self._emit_title(None)

        except Exception as e:
            QMessageBox.critical(self, "Midnight rollover error", str(e))

    def _restore_running_task(self):
        row = self.db.get_running_task()
        if not row:
            return

        name = row["name"]
        note = row["note"] or ""
        start_utc = iso_to_dt_utc(row["start_utc"])

        self.active_name = name
        self.active_note = note
        self.active_start = start_utc

        # show it as active
        self.active_label.setText(f"Active: {self.active_name}")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # set the elapsed to the real time since it started
        du = int((now_utc() - start_utc).total_seconds())
        if du < 0:
            du = 0  # in case of clock weirdness
        self.elapsed_label.setText(seconds_to_hms(du))

        # start ticking again
        self.timer.start()
        self._emit_title(du)


class HistoryTab(QWidget):
    def __init__(self, db: Database, timer_tab: TimerTab):
        super().__init__()
        self.db = db
        self.timer_tab = timer_tab

        main = QVBoxLayout(self)

        # Toolbar
        tb = QToolBar()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search name/note/date/time…")
        self.from_date = QDateEdit()
        self.from_date.setDisplayFormat("MM/dd/yyyy")
        self.from_date.setCalendarPopup(True)
        self.from_date.setSpecialValueText("From…")
        self.from_date.setDateRange(datetime(1970, 1, 1), datetime(2100, 1, 1))
        self.from_date.setDate(self.from_date.minimumDate())

        self.to_date = QDateEdit()
        self.to_date.setDisplayFormat("MM/dd/yyyy")
        self.to_date.setCalendarPopup(True)
        self.to_date.setSpecialValueText("To…")
        self.to_date.setDateRange(datetime(1970, 1, 1), datetime(2100, 1, 1))
        self.to_date.setDate(self.to_date.maximumDate())

        btn_refresh = QAction("Refresh", self)
        btn_edit = QAction("Edit Selected", self)
        btn_delete = QAction("Delete Selected", self)

        tb.addWidget(QLabel("Search:"))
        tb.addWidget(self.search_edit)
        tb.addSeparator()
        tb.addWidget(QLabel("From:"))
        tb.addWidget(self.from_date)
        tb.addWidget(QLabel("To:"))
        tb.addWidget(self.to_date)
        tb.addSeparator()
        tb.addAction(btn_refresh)
        tb.addAction(btn_edit)
        tb.addAction(btn_delete)
        main.addWidget(tb)

        # Table
        self.model = EntriesTableModel([])
        self.proxy = FilterProxy()
        self.proxy.setSourceModel(self.model)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(2, Qt.DescendingOrder)  # Start desc
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnHidden(self.proxy.mapFromSource(self.model.index(0, 0)).column(), True)  # hide ID via header after model load
        main.addWidget(self.table, 1)

        # Events
        self.search_edit.textChanged.connect(self.proxy.set_search_text)
        btn_refresh.triggered.connect(self.refresh)
        btn_edit.triggered.connect(self.edit_selected)
        btn_delete.triggered.connect(self.delete_selected)
        self.timer_tab.data_changed.connect(self.refresh)

        self.refresh()

    def _dates_filter(self):
        # convert QDateEdit to YYYY-MM-DD strings, unless special values
        dfrom = None
        dto = None
        min_date = self.from_date.minimumDate()
        max_date = self.to_date.maximumDate()
        if self.from_date.date() != min_date:
            dfrom = self.from_date.date().toString("yyyy-MM-dd")
        if self.to_date.date() != max_date:
            dto = self.to_date.date().toString("yyyy-MM-dd")
        return dfrom, dto

    def refresh(self):
        dfrom, dto = self._dates_filter()
        text = self.search_edit.text().strip()
        rows = self.db.search(text, dfrom, dto)
        self.model.set_rows(rows)
        # hide ID column
        self.table.setColumnHidden(0, True)

    def _selected_row(self) -> sqlite3.Row | None:
        idxs = self.table.selectionModel().selectedRows()
        if not idxs:
            return None
        proxy_idx = idxs[0]
        src_idx = self.proxy.mapToSource(proxy_idx)
        return self.model.row(src_idx.row())

    def edit_selected(self):
        row = self._selected_row()
        if not row:
            QMessageBox.information(self, "None", "Select a row to edit")
            return
        dlg = EditDialog(self.db, row, self)
        dlg.saved.connect(self.refresh)
        dlg.exec()

    def delete_selected(self):
        row = self._selected_row()
        if not row:
            QMessageBox.information(self, "None", "Select a row to delete")
            return
        rid = row["id"]
        if QMessageBox.question(self, "Confirm", f"Delete entry #{rid}?") == QMessageBox.Yes:
            self.db.delete_entries([rid])
            self.refresh()
            self.timer_tab.refresh_today()


# ---------- Main window ----------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Task Time Tracker")
        self.resize(1000, 700)

        # Style: Fusion with light dark-ish palette
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        self.setStyleSheet(
            """
            QWidget { font-size: 14px; }
            QTableView { gridline-color: #444; }
            QHeaderView::section { font-weight: 600; padding: 6px; }
            QPushButton { padding: 8px 14px; border-radius: 10px; }
            QLineEdit { padding: 6px 8px; }
            QToolBar { spacing: 8px; }
            """
        )

        self.db = Database(DB_FILE)

        tabs = QTabWidget()
        self.timer_tab = TimerTab(self.db)
        self.timer_tab.title_changed.connect(self.setWindowTitle)
        self.history_tab = HistoryTab(self.db, self.timer_tab)
        tabs.addTab(self.timer_tab, "Timer")
        tabs.addTab(self.history_tab, "History")

        self.setCentralWidget(tabs)


# ---------- Single-instance main ----------

def main():
    import sys

    # Per-user key so different users on the same machine can each run their own instance.
    singleton_key = f"time_tracker_{os.getuid()}"

    app = QApplication(sys.argv)

    # Try to connect to an existing instance.
    socket = QLocalSocket()
    socket.connectToServer(singleton_key)

    if socket.waitForConnected(150):
        # Existing instance is running: ask it to raise its window, then exit.
        socket.write(b"RAISE")
        socket.flush()
        socket.waitForBytesWritten(150)
        socket.disconnectFromServer()
        sys.exit(0)

    # No existing instance: clean up any stale server and become the primary instance.
    QLocalServer.removeServer(singleton_key)
    server = QLocalServer()
    if not server.listen(singleton_key):
        QMessageBox.critical(
            None,
            "Error",
            f"Could not start single-instance server '{singleton_key}'."
        )
        sys.exit(1)

    w = MainWindow()

    # Keep the server alive as long as the main window exists.
    w._singleton_server = server  # type: ignore[attr-defined]

    def handle_new_connection():
        conn = server.nextPendingConnection()
        if conn is None:
            return

        def on_ready_read():
            # Any message from a secondary instance means "raise the window".
            _ = conn.readAll()
            if w.isMinimized():
                w.showNormal()
            else:
                w.show()
            w.raise_()
            w.activateWindow()
            conn.disconnectFromServer()
            conn.deleteLater()

        conn.readyRead.connect(on_ready_read)

    server.newConnection.connect(handle_new_connection)

    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
