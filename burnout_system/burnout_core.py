"""
burnout_core.py — shared logic for the Burnout Risk dashboards.

Unchanged by the app.py merge. All three role-based views in app.py (Employee,
HR Manager, HR Analyst) import from here, so the model-loading, questionnaire
scoring, prediction and SQLite storage logic lives in ONE place (no duplication
across roles). This mirrors how a real system separates a service layer from
its user interfaces — now three interfaces, still one shared layer.
"""
import os
import io
import re
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths
# Anchored to THIS FILE's location, not the working directory. Previously these were relative
# paths that only resolved correctly when the apps were launched from the project root; now they
# resolve identically from anywhere. Same folder as before when run from root — no migration.
_HERE = os.path.dirname(os.path.abspath(__file__))   # .../FYP CODE/burnout_system
_ROOT = os.path.dirname(_HERE)                       # .../FYP CODE
_OUT  = os.path.join(_ROOT, "outputs")

ART     = os.path.join(_OUT, "model_artefacts")
DB_FILE = os.path.join(_OUT, "burnout_system.db")

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

# ------------------------------------------------------------------ artefact loading
def load_artefacts():
    """Load model, preprocessor, encoder and metadata. Returns a dict."""
    pre   = joblib.load(f"{ART}/preprocessor.joblib")
    model = joblib.load(f"{ART}/calibrated_model.joblib")
    le    = joblib.load(f"{ART}/label_encoder.joblib")
    meta  = joblib.load(f"{ART}/metadata.joblib")
    base  = joblib.load(f"{ART}/base_model.joblib")      if os.path.exists(f"{ART}/base_model.joblib") else None
    extra = joblib.load(f"{ART}/analytics_extra.joblib") if os.path.exists(f"{ART}/analytics_extra.joblib") else None
    return {"preprocessor": pre, "model": model, "label_encoder": le, "meta": meta,
            "base_model": base, "analytics_extra": extra}

def load_joblib(name):
    p = f"{ART}/{name}"
    return joblib.load(p) if os.path.exists(p) else None

def load_csv(name):
    p = f"{ART}/{name}"
    return pd.read_csv(p) if os.path.exists(p) else None

# ------------------------------------------------------------------ questionnaire spec (research-backed)
# Each item: (statement, reverse_scored). Responses 1..5 -> feature 0..1 via (mean-1)/4.
QUESTIONNAIRE = {
    "satisfaction_score": {
        "label": "Job Satisfaction",
        "source": "MOAQ Job Satisfaction Subscale (Cammann et al., 1983)",
        "anchors": ("Strongly disagree", "Strongly agree"),
        "items": [("All in all, I am satisfied with my job.", False),
                  ("In general, I do not like my job.", True),
                  ("In general, I like working here.", False)],
    },
    "workload_score": {
        "label": "Workload",
        "source": "Quantitative workload items, Karasek Job Content Instrument (Karasek, 1985)",
        "anchors": ("Never", "Always"),
        # The dataset's workload_score is a workload-HEALTH score (higher = more manageable),
        # not a workload-BURDEN score, despite its name. These validated items measure DEMAND
        # (higher = worse), so the scale score is inverted at the mapping step. See
        # score_features() for the evidence. Item wording is untouched.
        "maps_inverted": True,
        "items": [("I have too much work to do.", False),
                  ("I have to work very fast.", False),
                  ("I have to work extra hard to finish my tasks.", False),
                  ("I do not have enough time to get everything done.", False),
                  ("My workload feels unmanageable.", False)],
    },
    "collaboration_score": {
        "label": "Collaboration & Team Support",
        "source": "Coworker support, Job Content Questionnaire (Karasek et al., 1998)",
        "anchors": ("Strongly disagree", "Strongly agree"),
        "items": [("My coworkers are willing to help me get the job done.", False),
                  ("I can rely on my colleagues when things get difficult.", False),
                  ("There is good cooperation and teamwork in my team.", False),
                  ("People in my team share information and support each other.", False)],
    },
    "project_completion_rate": {
        "label": "Task & Project Completion",
        "source": "In-Role Behaviour scale (Williams & Anderson, 1991)",
        "anchors": ("Strongly disagree", "Strongly agree"),
        "items": [("I adequately complete the duties assigned to me.", False),
                  ("I meet the formal performance requirements of my role.", False),
                  ("I fulfil the responsibilities specified in my job description.", False),
                  ("I complete my projects and tasks on time.", False)],
    },
    "training_participation": {
        "label": "Development Opportunities",
        "source": "Growth-opportunities resource, JD-R Scale (Jackson & Rothmann, 2005)",
        "anchors": ("Strongly disagree", "Strongly agree"),
        "items": [("My job gives me opportunities to learn new skills.", False),
                  ("I have access to training and development in my role.", False),
                  ("My work allows me to grow professionally.", False),
                  ("I regularly take part in learning or development activities.", False)],
    },
    "career_progression_score": {
        "label": "Career Growth",
        "source": "Organizational Career Growth Scale (Weng et al., 2010)",
        "anchors": ("Strongly disagree", "Strongly agree"),
        "items": [("My current job moves me closer to my career goals.", False),
                  ("My job helps me develop new professional skills.", False),
                  ("My chances of being promoted here are good.", False),
                  ("The likelihood of my pay increasing here is high.", False),
                  ("I am making good progress toward my career goals.", False),
                  ("This job builds the skills I need to advance.", False)],
    },
}

# ------------------------------------------------------------------ organisational reference lists
# Canonical dropdown values, defined ONCE here and imported by both the app (employee sign-up)
# and populate_demo_data.py. Keeping a single source of truth is what makes the dropdowns
# consistent: every stored department/job_title is guaranteed to be one of these exact strings,
# so grouping and charting in the HR Manager dashboard never has to reconcile free-text variants
# ("Eng" vs "Engineering" vs "engineering"). Edit these lists to change the options everywhere.
DEPARTMENTS = ["Engineering", "IT", "Research & Development", "Sales & Marketing",
               "Customer Support", "Finance & Accounting", "Human Resources", "Operations",
               "Logistics & Supply Chain", "Administration", "Law & Compliance"]
JOB_TITLES  = ["Individual Contributor", "Senior Specialist", "Team Lead",
               "Manager", "Senior Manager", "Director"]

# ------------------------------------------------------------------ service class
class BurnoutService:
    """Wraps the model artefacts and exposes scoring / prediction / storage."""

    def __init__(self):
        a = load_artefacts()
        self.pre = a["preprocessor"]; self.model = a["model"]; self.le = a["label_encoder"]
        self.meta = a["meta"]; self.base = a["base_model"]
        self.S2 = self.meta["s2_features"]
        self.HIGH = self.meta["high_index"]
        self.CLASS_ORDER = self.meta["class_order"]
        self.MEDIANS = self.meta["feature_medians"]
        self.TIERS = self.meta["tier_thresholds"]
        self.ENGINEERED = self.meta.get("engineered_features", [])
        self.RAW = [f for f in self.S2 if f not in self.ENGINEERED]
        self.CAPS = self.meta.get("caps", {})
        self._explainer = None
        if SHAP_OK and self.base is not None:
            try:
                self._explainer = shap.TreeExplainer(self.base)
            except Exception:
                self._explainer = None

    # A bounded Likert scale cannot resolve the tails of a continuous population distribution:
    # four items on a five-point scale yield seventeen possible positions, so the top response
    # category means "the highest band this instrument can detect", NOT "the most extreme case in
    # 850,000 employees". Mapping it to p100 asserts the latter. In-role behaviour scales in
    # particular carry well-documented ceiling effects — almost every respondent endorses the top
    # categories — so an unbounded mapping systematically pushes ordinary respondents into the
    # population's extreme tail. Equating is therefore compressed onto [p5, p95].
    EQUATE_BOUNDS = (0.05, 0.95)

    # ---- questionnaire -> features
    def _equate(self, feat, pos01):
        """Equipercentile equating: a response at the p-th percentile of the questionnaire scale
        maps to the p-th percentile of that feature's TRAINING distribution. Matching
        percentile ranks across the two scales is standard equating practice (Kolen & Brennan,
        2014) and replaces an assumption the data contradicts with one the data supplies.
        """
        qs = (self.meta.get("feature_quantiles") or {}).get(feat)
        if not qs:
            return float(pos01)                      # fall back to the raw 0-1 position
        grid = self.meta.get("quantile_grid") or list(np.linspace(0, 1, len(qs)))
        lo, hi = self.EQUATE_BOUNDS
        p = lo + float(pos01) * (hi - lo)          # response position -> bounded percentile
        return float(np.interp(p, grid, qs))

    def score_features(self, responses, overtime_hours, s2_features):
        """Questionnaire responses -> model features, on the scale the model was trained on.

        SEMANTIC ALIGNMENT (`maps_inverted`). The dataset's `workload_score` measures workload
        HEALTH, not workload BURDEN, despite its name. Three independent lines of evidence:

          1. corr(workload_score, overtime_hours) = -0.616 — the MORE overtime an employee works,
             the LOWER their workload_score. A burden measure would correlate positively; this is
             the strongest relationship the column has, and it settles the direction.
          2. corr(workload_score, stress_level) = -0.410 and corr(workload_score, burnout_risk)
             = -0.404 — a higher score accompanies LESS stress and LESS burnout.
          3. Every other `_score` column in this dataset is coded higher = better (satisfaction,
             collaboration, performance, project_completion, career_progression) and every one of
             them correlates negatively with burnout. workload_score sits inside that group.

        The questionnaire uses Karasek's quantitative DEMAND items ("I have too much work to do"),
        where a high response means high burden. Mapping those directly onto a health-coded target
        feature inverts the construct, which is what produced the nonsensical predictions
        this alignment fixes. Only the SCORING KEY is aligned to the dataset's coding; the
        validated item wording is unchanged, so the instrument itself is untouched. This is a unit
        conversion between two codings of one construct, not a change to what is being measured.
        """
        feats = {}
        for feat, spec in QUESTIONNAIRE.items():
            vals = responses[feat]
            adj = [(6 - v if rev else v) for v, (_, rev) in zip(vals, spec["items"])]
            pos = (float(np.mean(adj)) - 1) / 4            # 1..5 Likert -> 0..1 POSITION on the scale
            if spec.get("maps_inverted"):                  # demand-coded item -> health-coded feature
                pos = 1.0 - pos
            feats[feat] = round(self._equate(feat, pos), 4)   # position -> value on the model's scale
        feats["overtime_hours"] = float(overtime_hours)
        if "workload_dissatisfaction" in s2_features:
            # (1 - workload_health) = workload BURDEN; (1 - satisfaction) = dissatisfaction.
            # Must match add_engineered_features() in the notebook EXACTLY - this term is
            # recomputed at inference, so any divergence silently feeds the model a feature it
            # was never trained on.
            feats["workload_dissatisfaction"] = round(
                (1 - feats["workload_score"]) * (1 - feats["satisfaction_score"]), 4)
        return feats

    def features_for(self, responses, overtime_hours):
        # score_features is now an instance method (it needs the training quantiles for equating)
        return self.score_features(responses, overtime_hours, self.S2)

    # ---- preprocessing helpers
    def _apply_caps(self, d):
        d = d.copy()
        for col, cap in (self.CAPS or {}).items():
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce").clip(upper=cap)
        return d

    def _engineer(self, d):
        d = d.copy()
        for c in self.RAW:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
                if d[c].isna().any():
                    d[c] = d[c].fillna(self.MEDIANS.get(c, 0))
        d = self._apply_caps(d)
        if "workload_dissatisfaction" in self.S2 and {"workload_score", "satisfaction_score"}.issubset(d.columns):
            # Mirrors add_engineered_features() in the notebook and score_features() above.
            d["workload_dissatisfaction"] = (1 - d["workload_score"]) * (1 - d["satisfaction_score"])
        return d

    def tier(self, p_high):
        return "Priority" if p_high >= self.TIERS["Priority"] else ("Elevated" if p_high >= self.TIERS["Elevated"] else "Monitor")

    # ---- prediction
    def predict(self, feat_df):
        d = self._engineer(feat_df)
        X = d[self.S2]
        Xp = self.pre.transform(X)
        proba = self.model.predict_proba(Xp)
        p_high = proba[:, self.HIGH]
        pred = self.le.inverse_transform(proba.argmax(axis=1))
        tiers = [self.tier(p) for p in p_high]
        return {"Xp": Xp, "proba": proba, "p_high": p_high, "pred": pred, "tiers": tiers}

    # ---- SHAP
    def shap_high(self, Xp):
        if self._explainer is None:
            return None, None
        try:
            sv = self._explainer.shap_values(Xp)
            if isinstance(sv, list):
                sh = np.asarray(sv[self.HIGH])
            elif np.ndim(sv) == 3:
                sh = np.asarray(sv)[:, :, self.HIGH]
            else:
                sh = np.asarray(sv)
            names = [n.split("__", 1)[-1] for n in self.pre.get_feature_names_out()]
            return names, sh
        except Exception:
            return None, None

    @staticmethod
    def top_drivers_text(row, names, k=3):
        order = np.argsort(row)[::-1]
        picks = [names[i] for i in order[:k] if row[i] > 0]
        return "; ".join(picks) if picks else "no strong upward drivers"

# ------------------------------------------------------------------ storage (SQLite)
import sqlite3
import hashlib
from contextlib import contextmanager

SCHEMA_VERSION = 1

# Bumped whenever the questionnaire -> feature mapping changes in a way that alters predictions.
#   1 = linear (mean-1)/4 mapping, workload demand-coded  [SUPERSEDED - produced inverted results]
#   2 = workload semantically aligned to the dataset's health coding; equipercentile equating
#       onto the training distribution; equating bounded to [p5, p95]
#   3 = workload_dissatisfaction corrected to (1 - workload_health) * (1 - satisfaction), i.e.
#       BURDEN x dissatisfaction. Requires the model retrained on the corrected feature.
SCORING_VERSION = "3"

import hmac
# PBKDF2 with 200k rounds is deliberately slow, which is the point: it makes brute-forcing a
# stolen hash expensive. Passwords are NEVER stored or logged in plaintext, and never leave this
# module. This demonstrates access separation for a proof of concept — it is not production
# authentication (no rate limiting, no password policy, no reset flow, no session expiry).
PBKDF2_ROUNDS = 200_000

def _hash_password(password, salt=None):
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return salt.hex(), dk.hex()

def _verify_password(password, salt_hex, hash_hex):
    if not salt_hex or not hash_hex:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             bytes.fromhex(salt_hex), PBKDF2_ROUNDS)
    return hmac.compare_digest(dk.hex(), hash_hex)   # timing-safe: a plain == leaks length info

@contextmanager
def _connect():
    """One connection per operation, committed or rolled back, then closed.

    check_same_thread=False is required because Streamlit re-executes the script in worker
    threads: a connection created during one rerun would be rejected on the next. Opening per
    operation (rather than caching one global connection) keeps threads isolated entirely.
    timeout=10 lets a write wait rather than fail if the other app holds the lock.
    """
    con = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=10)
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

def _feature_columns():
    """The schema's feature columns are read from the deployed model's own metadata, so the
    table cannot drift out of sync with the model's expected inputs."""
    meta = load_joblib("metadata.joblib") or {}
    return list(meta.get("s2_features", []))

_MODEL_VERSION = None
def model_version():
    """Identifies WHICH model produced a stored prediction.

    Without this the stored prediction columns are unattributable: you would know what the
    dashboard said but not what said it. The fingerprint covers the model artefact, the metadata
    it depends on, and the scoring-key version, so it changes automatically whenever any part of
    the inference path is altered.
    """
    global _MODEL_VERSION
    if _MODEL_VERSION is not None:
        return _MODEL_VERSION
    try:
        meta = load_joblib("metadata.joblib") or {}
        name = f"{meta.get('model_name', 'model')}+{meta.get('strategy', 'na')}+isotonic"
        # Fingerprint the whole inference contract, not just the estimator. A stored prediction is
        # reproducible only if the model, the metadata it depends on (winsorisation caps, training
        # medians, the equating quantiles) AND the scoring code that produced its inputs are all
        # identified. Versioning the estimator alone is not an audit trail: this project's most
        # serious defect to date lived in the mapping layer and left the model file untouched, so
        # a model-only fingerprint would have marked broken and corrected predictions identically.
        h = hashlib.md5()
        for fn in ("calibrated_model.joblib", "metadata.joblib"):
            p = os.path.join(ART, fn)
            if os.path.exists(p):
                h.update(open(p, "rb").read())
        _MODEL_VERSION = f"{name}@{h.hexdigest()[:8]}/s{SCORING_VERSION}"
    except Exception:
        _MODEL_VERSION = "unknown"
    return _MODEL_VERSION

def init_db():
    """Create the schema if absent. Safe to call on every operation."""
    feats = _feature_columns()
    if not feats:
        raise RuntimeError("metadata.joblib not found or has no s2_features — "
                           "run the notebook's export cells before starting the apps.")
    os.makedirs(_OUT, exist_ok=True)
    feat_ddl = ",\n            ".join(f"[{c}] REAL" for c in feats)
    with _connect() as con:
        # employees: one row per person. employee_name lives HERE, not on submissions, because it
        # depends on employee_id rather than on a submission — storing it per-submission would be
        # a transitive dependency and break third normal form.
        con.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id   TEXT PRIMARY KEY,
            employee_name TEXT,
            department    TEXT,
            job_title     TEXT,
            first_seen    TEXT NOT NULL
        )""")
        # submissions: one row per check-in. The prediction columns are a deliberate, documented
        # denormalisation - they are derivable from the features plus the model, but are stored as
        # an immutable audit record of what the dashboard told HR at that moment, tagged with the
        # model_version that produced them. Recomputing on read would silently rewrite history.
        con.execute(f"""
        CREATE TABLE IF NOT EXISTS submissions (
            submission_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id     TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            {feat_ddl},
            predicted_class TEXT NOT NULL,
            prob_high       REAL NOT NULL CHECK (prob_high BETWEEN 0 AND 1),
            risk_tier       TEXT NOT NULL CHECK (risk_tier IN ('Priority','Elevated','Monitor')),
            model_version   TEXT NOT NULL,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
        )""")

        # Migration: CREATE TABLE IF NOT EXISTS will not add columns to a database that already
        # exists, so the credential columns are added explicitly. Existing employees keep their
        # check-in history and simply have no account until they register.
        _cols = {r[1] for r in con.execute("PRAGMA table_info(employees)").fetchall()}
        if "password_salt" not in _cols:
            con.execute("ALTER TABLE employees ADD COLUMN password_salt TEXT")
        if "password_hash" not in _cols:
            con.execute("ALTER TABLE employees ADD COLUMN password_hash TEXT")
        # Same pattern for the ERD's new employee attributes: existing databases keep their rows
        # and simply carry NULL department/job_title until the employee (re-)registers.
        if "department" not in _cols:
            con.execute("ALTER TABLE employees ADD COLUMN department TEXT")
        if "job_title" not in _cols:
            con.execute("ALTER TABLE employees ADD COLUMN job_title TEXT")

        con.execute("CREATE INDEX IF NOT EXISTS ix_sub_employee  ON submissions(employee_id)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_sub_timestamp ON submissions(timestamp)")

def save_submission(record):
    """INSERT one check-in. Upserts the employee, then writes the submission + prediction."""
    init_db()
    feats = _feature_columns()
    eid  = str(record.get("employee_id", "") or "").strip()
    name = str(record.get("employee_name", "") or "").strip()
    ts   = record.get("timestamp") or now_str()
    if not eid:
        raise ValueError("employee_id is required.")

    cols = (["employee_id", "timestamp"] + feats
            + ["predicted_class", "prob_high", "risk_tier", "model_version"])
    vals = ([eid, ts]
            + [None if record.get(c) is None else float(record.get(c)) for c in feats]
            + [str(record.get("predicted_class")), float(record.get("prob_high")),
               str(record.get("risk_tier")), model_version()])

    with _connect() as con:
        con.execute("INSERT OR IGNORE INTO employees(employee_id, employee_name, first_seen) "
                    "VALUES (?,?,?)", (eid, name, ts))
        if name:   # keep the latest non-blank name without clobbering first_seen
            con.execute("UPDATE employees SET employee_name=? WHERE employee_id=?", (name, eid))
        con.execute(f"INSERT INTO submissions ({', '.join('[%s]' % c for c in cols)}) "
                    f"VALUES ({', '.join('?' * len(cols))})", vals)

def load_submissions():
    """SELECT all check-ins joined to employee names -> DataFrame.

    Returns the same column shape the CSV did, so both apps are unchanged by the storage swap.
    """
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
    try:
        feats = _feature_columns()
        if not feats:
            return pd.DataFrame()
        sel = ", ".join(f"s.[{c}]" for c in feats)
        sql = f"""
            SELECT s.submission_id,
                   s.employee_id,
                   COALESCE(e.employee_name, '') AS employee_name,
                   COALESCE(e.department, '')    AS department,
                   COALESCE(e.job_title, '')     AS job_title,
                   s.timestamp,
                   {sel},
                   s.predicted_class, s.prob_high, s.risk_tier, s.model_version
            FROM submissions s
            LEFT JOIN employees e ON e.employee_id = s.employee_id
            ORDER BY s.timestamp DESC, s.submission_id DESC
        """
        with _connect() as con:
            return pd.read_sql_query(sql, con)
    except Exception:
        return pd.DataFrame()

def clear_submissions():
    """DELETE all rows, keeping the schema. FK cascade is redundant here but left enabled."""
    if not os.path.exists(DB_FILE):
        return
    with _connect() as con:
        con.execute("DELETE FROM submissions")
        con.execute("DELETE FROM employees")
        con.execute("DELETE FROM sqlite_sequence WHERE name='submissions'")

# ------------------------------------------------------------------ employee accounts
def employee_account(eid):
    """-> (exists, has_account, name). Distinguishes 'no such ID' from 'ID with no password'."""
    init_db()
    with _connect() as con:
        r = con.execute("SELECT employee_name, password_hash FROM employees WHERE employee_id=?",
                        (str(eid).strip(),)).fetchone()
    if r is None:
        return False, False, ""
    return True, bool(r[1]), (r[0] or "")

def register_employee(eid, name, password, department=None, job_title=None):
    """Create an account, or claim an existing employee_id that has no password yet.

    department and job_title are optional so any older caller that omits them keeps working; when
    supplied they are stored on the employees table (the ERD attributes). On the "claim existing
    ID" branch they are written with COALESCE(NULLIF(...)) so a blank value never wipes a
    department/title that a previous check-in or registration already recorded.
    """
    eid, name = str(eid).strip(), str(name or "").strip()
    dept  = str(department or "").strip()
    title = str(job_title or "").strip()
    if not eid:
        return False, "Please enter your employee ID."
    if not re.fullmatch(r"E\d{4}", eid):
        return False, "Employee ID must be in the format E#### (E followed by four digits)."
    if len(password or "") < 6:
        return False, "Please choose a password of at least 6 characters."
    exists, has_acct, _ = employee_account(eid)
    if has_acct:
        return False, "An account already exists for this ID. Please sign in instead."
    salt, digest = _hash_password(password)
    with _connect() as con:
        if exists:
            # An ID from the existing records: claim it, keeping its check-in history intact.
            con.execute("UPDATE employees SET password_salt=?, password_hash=?, "
                        "employee_name=COALESCE(NULLIF(?,''), employee_name), "
                        "department=COALESCE(NULLIF(?,''), department), "
                        "job_title=COALESCE(NULLIF(?,''), job_title) WHERE employee_id=?",
                        (salt, digest, name, dept, title, eid))
        else:
            con.execute("INSERT INTO employees(employee_id, employee_name, department, job_title, "
                        "first_seen, password_salt, password_hash) VALUES (?,?,?,?,?,?,?)",
                        (eid, name, dept, title, now_str(), salt, digest))
    return True, "Account created."

def authenticate_employee(eid, password):
    """-> (ok, name_or_error). Deliberately does not reveal whether the ID exists."""
    eid = str(eid).strip()
    with _connect() as con:
        r = con.execute("SELECT employee_name, password_salt, password_hash FROM employees "
                        "WHERE employee_id=?", (eid,)).fetchone()
   
    if r is None or not _verify_password(password or "", r[1], r[2]):
        return False, "Incorrect employee ID or password."
    return True, (r[0] or "")

def employee_history_summary(eid):
    """-> (n_check_ins, last_timestamp). Counts only — no risk score is ever returned to an
    employee, by design."""
    if not os.path.exists(DB_FILE):
        return 0, None
    with _connect() as con:
        r = con.execute("SELECT COUNT(*), MAX(timestamp) FROM submissions WHERE employee_id=?",
                        (str(eid).strip(),)).fetchone()
    return (int(r[0]) if r else 0), (r[1] if r else None)


def db_schema_text():
    """The live DDL, straight from the database - use this for the report's data dictionary
    rather than transcribing it by hand."""
    if not os.path.exists(DB_FILE):
        return "(database not created yet)"
    with _connect() as con:
        rows = con.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
                           "ORDER BY type DESC, name").fetchall()
    return "\n\n".join(r[0] for r in rows)

def to_excel_bytes(df):
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="BurnoutScores")
        return buf.getvalue()
    except Exception:
        return None

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
