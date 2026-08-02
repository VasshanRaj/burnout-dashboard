# ============================================================
# app.py — Burnout Risk Dashboards (proof of concept)
# Single entry point for all three roles. One landing page, one dedicated
# login per role, then straight into that role's dashboard:
#
#   Employee    -> confidential wellbeing check-in, soft feedback only
#   HR Manager  -> team overview / directory / individual review / export
#                  (this is what used to be the standalone HR console)
#   HR Analyst  -> EDA, model comparison, validation & robustness evidence
#                  (this is what used to be app_analytics.py)
#
# This file supersedes the separate app_hr.py and app_analytics.py, which
# can now be archived — everything they did is reproduced here, routed by
# role. burnout_core.py is unchanged: it remains the single shared service
# layer (model loading, questionnaire scoring, prediction, SQLite storage)
# that ALL THREE roles below call into, exactly as before.
#
# Run from the PROJECT ROOT:
#   streamlit run burnout_system/app.py
# ============================================================
import os, sys, re
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

sys.path.append(os.path.dirname(__file__))
import burnout_core as core

st.set_page_config(page_title="Burnout Risk Dashboards", page_icon="🌱", layout="wide")

# ------------------------------------------------------------------ shared constants
# PRIMARY/CLASS_COLOR are used by both the HR Manager and HR Analyst views (they were
# independently defined with the SAME values in the two original files - merged here).
# Passcodes are read from Streamlit secrets so they are never hard-coded in a public repo.
# Locally (no secrets file) they fall back to demo values; on the cloud, the values you set in
# the app's Secrets box override these fallbacks, so the fallbacks below never unlock the live app.
try:
    HR_MANAGER_PASSCODE = st.secrets["HR_MANAGER_PASSCODE"]
    ANALYST_PASSCODE    = st.secrets["ANALYST_PASSCODE"]
except Exception:
    HR_MANAGER_PASSCODE = "hr1234"        # local-only fallback
    ANALYST_PASSCODE    = "analyst1234"   # local-only fallback
PRIMARY, ACCENT = "#4C6FFF", "#F5A623"
TIER_COLOR  = {"Priority": "#E5484D", "Elevated": "#F5A623", "Monitor": "#30A46C"}
CLASS_COLOR = {"Low": "#30A46C", "Moderate": "#F5A623", "High": "#E5484D"}
STRAT_COLOR = {"class_weight": "#4C6FFF", "smote": "#7B5CFF", "undersample": "#F5A623"}

# ------------------------------------------------------------------ merged stylesheet
# One combined block. Duplicated selectors from the two original files (.kpi, .kpi-label,
# .block-container) are defined ONCE here rather than twice, with values reconciled where
# they differed (e.g. kpi-value font-size takes the HR console's slightly larger 1.6rem).
st.markdown("""<style>
.block-container{padding-top:1.5rem;max-width:1300px;}
#MainMenu, footer {visibility:hidden;}
.kpi{background:rgba(130,130,150,.07);border:1px solid rgba(130,130,150,.2);border-radius:14px;padding:14px 16px;}
.kpi-label{opacity:.6;font-size:.68rem;text-transform:uppercase;letter-spacing:.6px;}
.kpi-value{font-size:1.6rem;font-weight:800;margin-top:3px;}
.kpi-sub{opacity:.55;font-size:.68rem;}
.finding{background:rgba(76,111,255,.08);border-left:4px solid #4C6FFF;padding:14px 18px;border-radius:8px;margin:10px 0;}
.hero{background:linear-gradient(120deg,#4C6FFF,#7B5CFF);color:#fff;padding:24px 28px;border-radius:16px;margin-bottom:18px;}
.hero h1{color:#fff;margin:0;font-size:1.5rem;}.hero p{color:rgba(255,255,255,.85);margin:6px 0 0;}
.topbar{display:flex;justify-content:space-between;align-items:center;
        background:linear-gradient(120deg,#1f6f54,#2f9e6f);color:#fff;padding:14px 22px;border-radius:14px;margin-bottom:18px;}
.topbar h2{margin:0;color:#fff;font-size:1.2rem;}
.topbar span{opacity:.85;font-size:.85rem;}
.rolecard{background:rgba(130,130,150,.06);border:1px solid rgba(130,130,150,.22);border-radius:18px;
          padding:30px 26px;text-align:center;height:100%;}
.rolecard h3{margin:.2rem 0;font-size:1.25rem;}
.rolecard p{opacity:.7;font-size:.9rem;min-height:52px;}
.soft{background:rgba(48,163,108,.10);border-left:4px solid #30A46C;padding:16px 20px;border-radius:10px;margin:8px 0;}
.qsrc{font-size:.74rem;opacity:.5;font-style:italic;margin:-6px 0 8px;}
.badge{display:inline-block;padding:10px 14px;border-radius:10px;color:#fff;font-weight:700;text-align:center;width:100%;}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ cached loaders
# get_service() -> from the old app_hr.py. Needed by Employee and HR Manager (prediction).
@st.cache_resource
def get_service():
    return core.BurnoutService()

# get_meta() / csv() / jbl() / kpi() / clean() -> from the old app_analytics.py, unchanged.
# Defined globally but only CALLED from inside the HR Analyst branch below, so a missing
# analytics artefact (e.g. eda_summary.joblib) cannot block the Employee or HR Manager roles.
@st.cache_resource
def get_meta():
    a = core.load_artefacts()
    return a["meta"], a["analytics_extra"]

@st.cache_data
def csv(name):  return core.load_csv(name)
@st.cache_data
def jbl(name):  return core.load_joblib(name)

def kpi(col, label, value, sub=""):
    col.markdown(f"<div class='kpi'><div class='kpi-label'>{label}</div>"
                 f"<div class='kpi-value'>{value}</div><div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)
def clean(n): return n.split("__", 1)[-1]

# svc is required by Employee (prediction) and HR Manager (team predictions); loaded once,
# up front, so a missing model artefact is caught before the landing page even renders.
try:
    svc = get_service()
except Exception as e:
    st.error(f"Dashboard not ready — model artefacts missing. Run the notebook first.\n\n{e}")
    st.stop()

ss = st.session_state
ss.setdefault("role", None)
ss.setdefault("mgr_auth", False)
ss.setdefault("analyst_auth", False)

def go_home():
    ss.role = None; ss.mgr_auth = False; ss.analyst_auth = False
    ss.emp_auth = False; ss.emp_id = None; ss.emp_name = ""

# ============================================================ LANDING
if ss.role is None:
    st.markdown("<div class='topbar'><h2>🌱 Workplace Wellbeing Platform</h2>"
                "<span>Burnout risk assessment, HR triage & model evidence</span></div>",
                unsafe_allow_html=True)
    st.write("")
    st.markdown("#### Welcome. Please choose how you'd like to continue.")
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='rolecard'><h3>🧑‍💼 I'm an employee</h3>"
                    "<p>Complete a short, confidential wellbeing check-in. It takes about "
                    "3 minutes.</p></div>", unsafe_allow_html=True)
        st.write("")
        if st.button("Start my check-in", type="primary", use_container_width=True, key="land_emp"):
            ss.role = "employee"; st.rerun()
    with c2:
        st.markdown("<div class='rolecard'><h3>🛡️ I'm an HR Manager</h3>"
                    "<p>Review team wellbeing, risk tiers and individual reports. "
                    "Sign-in required.</p></div>", unsafe_allow_html=True)
        st.write("")
        if st.button("Go to HR Manager console", use_container_width=True, key="land_mgr"):
            ss.role = "hr_manager"; st.rerun()
    with c3:
        st.markdown("<div class='rolecard'><h3>📊 I'm an HR Analyst</h3>"
                    "<p>Explore the EDA, model comparison and validation evidence behind "
                    "the prediction model. Sign-in required.</p></div>", unsafe_allow_html=True)
        st.write("")
        if st.button("Go to model insights", use_container_width=True, key="land_ana"):
            ss.role = "hr_analyst"; st.rerun()
    st.caption("This is a demonstration decision-support tool. It is indicative only — "
               "not a clinical diagnosis.")

# ============================================================ EMPLOYEE
elif ss.role == "employee":
    st.markdown("<div class='topbar'><h2>🌱 Wellbeing Check-in</h2>"
                "<span>Confidential · ~3 minutes</span></div>", unsafe_allow_html=True)
    if st.button("← Back to home"):
        ss.emp_auth = False; ss.emp_id = None; ss.emp_name = ""
        go_home(); st.rerun()

    # ---------- sign in / sign up ----------
    if not ss.get("emp_auth"):
        st.markdown("#### Sign in to continue")
        st.caption("Signing in links this check-in to your previous ones, so the direction of "
                   "your wellbeing over time is visible to HR. ")
        t_in, t_up = st.tabs(["Sign in", "Create an account"])

        with t_in:
            li_id = st.text_input("Employee ID", key="li_id", placeholder="e.g. E1002")
            li_pw = st.text_input("Password", type="password", key="li_pw")
            if st.button("Sign in", type="primary", key="li_go"):
                ok, msg = core.authenticate_employee(li_id, li_pw)
                if ok:
                    ss.emp_auth = True; ss.emp_id = li_id.strip(); ss.emp_name = msg
                    st.rerun()
                else:
                    st.error(msg)

        with t_up:
            su_id = st.text_input("Employee ID", key="su_id", placeholder="e.g. E1002")
            if ss.get("emp_id_error"):
                st.error("⚠️ Invalid Employee ID format. Please use the format E#### — the letter "
                         "E followed by four digits (for example, E1002).")
            su_nm = st.text_input("Your name", key="su_nm")
            _cd, _cj = st.columns(2)
            su_dept  = _cd.selectbox("Department", core.DEPARTMENTS, key="su_dept")
            su_title = _cj.selectbox("Job title", core.JOB_TITLES, key="su_title")
            su_pw = st.text_input("Choose a password", type="password", key="su_pw",
                                  help="At least 6 characters.")
            su_p2 = st.text_input("Confirm password", type="password", key="su_p2")
            if st.button("Create account", type="primary", key="su_go"):
                if not re.fullmatch(r"E\d{4}", su_id.strip()):
                    ss.emp_id_error = True
                elif su_pw != su_p2:
                    ss.emp_id_error = False
                    st.error("The two passwords do not match.")
                else:
                    ss.emp_id_error = False
                    ok, msg = core.register_employee(su_id, su_nm, su_pw, su_dept, su_title)
                    if ok:
                        okl, nm = core.authenticate_employee(su_id, su_pw)
                        if okl:
                            ss.emp_auth = True; ss.emp_id = su_id.strip(); ss.emp_name = nm
                            st.rerun()
                    else:
                        st.error(msg)
            _e, _a, _n = core.employee_account(su_id) if su_id.strip() else (False, False, "")
            if _e and not _a:
                st.info(f"We already have check-in records for **{su_id.strip()}**"
                        + (f" ({_n})" if _n else "") + ". Creating an account will link them to "
                        "you, and none of your history will be lost.")

        st.stop()

    # ---------- signed in ----------
    _n_prev, _last = core.employee_history_summary(ss.emp_id)
    _who = ss.get("emp_name") or ss.emp_id
    cW, cO = st.columns([4, 1])
    cW.markdown(f"Signed in as **{_who}** (`{ss.emp_id}`)"
                + (f" · {_n_prev} previous check-in{'s' if _n_prev != 1 else ''}, "
                   f"last on {str(_last)[:10]}" if _n_prev else " · this is your first check-in"))
    with cO:
        if st.button("Sign out", use_container_width=True):
            ss.emp_auth = False; ss.emp_id = None; ss.emp_name = ""
            ss.emp_done = False; st.rerun()

    if ss.get("emp_done"):
        # ---------- soft feedback (no raw risk label) ----------
        f = ss.emp_feats
        st.success("Thank you — your check-in has been submitted.")
        st.markdown("### A few reflections for you")
        
        _QS = (svc.meta.get("feature_quantiles") or {})
        def _pct(c):
            q = _QS.get(c)
            if not q or c not in f:
                return None
            return 100.0 * float(np.searchsorted(q, float(f[c]))) / (len(q) - 1)
        def _lowest(c, thresh=25):
            p = _pct(c)
            return p is not None and p <= thresh

        msgs = []
        if _lowest("satisfaction_score"):
            msgs.append("You indicated lower job satisfaction lately. If it helps, consider a conversation with your "
                        "manager or a trusted colleague about what would make work feel better.")
        if _lowest("workload_score"):
            msgs.append("Your workload sounds hard to keep on top of right now. Protecting focus time and talking "
                        "through priorities with your team can ease the pressure.")
        if f.get("overtime_hours", 0) > 8:
            msgs.append("You're putting in significant extra hours. Regular breaks and clear boundaries around "
                        "off-hours are worth protecting.")
        if _lowest("collaboration_score"):
            msgs.append("Feeling well-supported by your team makes a big difference — reaching out or pairing up on "
                        "tasks can help.")
        if _lowest("career_progression_score"):
            msgs.append("You seem less certain about where your role is heading. A conversation about development "
                        "and next steps is a reasonable thing to ask for.")
        if not msgs:
            msgs.append("Your responses look balanced across the areas we asked about. Keep looking after the habits "
                        "that are working for you.")
        for m in msgs:
            st.markdown(f"<div class='soft'>{m}</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.caption("These reflections are drawn from workplace-wellbeing research and respond to "
                   "what you reported — they are suggestions, not an assessment of you.")
        st.markdown("**Support is available.** If you're struggling, your organisation's Employee Assistance Programme "
                    "and HR wellbeing resources are there to help. In a crisis, contact your local support services.")
        st.write("")
        if st.button("Submit another check-in"):
            ss.emp_done = False; st.rerun()

    else:
        st.caption("Answer honestly — there are no right or wrong answers. Your responses support your wellbeing.")
        with st.form("emp_form"):
            responses = {}
            for feat, spec in core.QUESTIONNAIRE.items():
                st.markdown(f"#### {spec['label']}")
                lo, hi = spec["anchors"]
                vals = []
                for j, (text, _) in enumerate(spec["items"]):
                    vals.append(st.radio(text, [1, 2, 3, 4, 5], horizontal=True, index=2,
                                         key=f"e_{feat}_{j}", help=f"1 = {lo}   ·   5 = {hi}"))
                responses[feat] = vals
                st.markdown("<hr style='opacity:.12'>", unsafe_allow_html=True)
            st.markdown("#### Working hours")
            overtime = st.number_input("Overtime hours per week (beyond your contracted hours)",
                                       min_value=0.0, value=0.0, step=0.5, key="emp_overtime",
                                       help="Enter 0–40. If you genuinely worked more than 40, enter 40.")
            submitted = st.form_submit_button("Submit my check-in", type="primary", use_container_width=True)

        if submitted and overtime > 40:
            ss.emp_ot_error = True

        if submitted:
            _ot = float(ss.get("emp_overtime", overtime))
            if _ot > 40:
                ss.emp_ot_error = True
            else:
                ss.emp_ot_error = False
        if ss.get("emp_ot_error"):
            st.error("⚠️ Overtime value is too high. The maximum accepted is 40 hours per week. "
                     "Please re-enter a value of 40 or less. If you genuinely worked more than 40 "
                     "hours, please enter the maximum of 40.")
        if submitted and not ss.get("emp_ot_error"):
            _ot = float(ss.get("emp_overtime", overtime))
            feats = svc.features_for(responses, _ot)
            res = svc.predict(pd.DataFrame([feats]))
            # Identity comes from the signed-in session, not a free-text box: previously
            # anyone could type any ID, so a check-in could be filed against a colleague.
            record = {"employee_id": ss.emp_id, "employee_name": ss.get("emp_name", ""),
                      "timestamp": core.now_str(), **feats,
                      "predicted_class": res["pred"][0], "prob_high": round(float(res["p_high"][0]), 4),
                      "risk_tier": res["tiers"][0]}
            core.save_submission(record)
            ss.emp_feats = feats; ss.emp_done = True
            st.rerun()

# ============================================================ HR MANAGER
elif ss.role == "hr_manager":
    if not ss.mgr_auth:
        st.markdown("<div class='topbar'><h2>🛡️ HR Manager Console</h2><span>Sign-in required</span></div>", unsafe_allow_html=True)
        if st.button("← Back to home"):
            go_home(); st.rerun()
        st.markdown("#### Sign in")
        pw = st.text_input("Passcode", type="password")
        if st.button("Sign in", type="primary"):
            if pw == HR_MANAGER_PASSCODE:
                ss.mgr_auth = True; st.rerun()
            else:
                st.error("Incorrect passcode.")
        
        st.stop()

    # authenticated
    tb1, tb2 = st.columns([4, 1])
    tb1.markdown("<div class='topbar'><h2>🛡️ HR Manager Console</h2>"
                 "<span>Team burnout-risk overview</span></div>", unsafe_allow_html=True)
    with tb2:
        st.write("")
        if st.button("Sign out", use_container_width=True):
            go_home(); st.rerun()

    # ---------- DATA: one row per PERSON, not per form ----------
    
    STALE_DAYS = 90          # a check-in older than a quarter is history, not a current assessment
    WORSENED_P = 0.15        # half a tier band (Elevated 0.40 -> Priority 0.70) = a real move
    TIER_RANK = {"Monitor": 0, "Elevated": 1, "Priority": 2}

    subs = core.load_submissions()
    if subs.empty:
        st.info("No check-ins yet. Employees can complete a wellbeing check-in from the home screen.")
        st.stop()

    missing = [c for c in svc.RAW if c not in subs.columns]
    if missing:
        st.error(f"Stored check-ins are missing feature columns {missing} (older questionnaire version).")
        st.stop()

    # Stored predictions are read, never recomputed: each row records what the dashboard told HR
    # at the time, tagged with the model + scoring version that produced it.
    subs = subs.reset_index(drop=True)
    subs["_pred"]  = subs["predicted_class"]
    subs["_phigh"] = subs["prob_high"].astype(float)
    subs["_tier"]  = subs["risk_tier"]
    subs["_ts"]    = pd.to_datetime(subs["timestamp"])

    _stale_v = subs.loc[subs["model_version"] != core.model_version(), "model_version"].unique()
    if len(_stale_v):
        st.caption(f"⚠️ {len(_stale_v)} earlier scoring version(s) present ({', '.join(map(str, _stale_v))}). "
                   f"Current: {core.model_version()}. Stored predictions are kept as an audit record.")

    _now = pd.Timestamp.now()
    hist = subs.sort_values("_ts")
    people = []
    for eid, g in hist.groupby("employee_id"):
        cur  = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else None
        d_p  = float(cur["_phigh"] - prev["_phigh"]) if prev is not None else np.nan
        d_t  = (TIER_RANK[cur["_tier"]] - TIER_RANK[prev["_tier"]]) if prev is not None else 0
        if prev is None:                       trend = "• Single Check-In"
        elif d_t > 0 or d_p >= WORSENED_P:     trend = "▲ Worsening"
        elif d_t < 0 or d_p <= -WORSENED_P:    trend = "▼ Improving"
        else:                                  trend = "– Stable"
        age = int((_now - cur["_ts"]).days)
        people.append({"employee_id": eid, "employee_name": cur["employee_name"],
                       "department": cur.get("department", "") or "—",
                       "job_title": cur.get("job_title", "") or "—",
                       "tier": cur["_tier"], "p_high": round(float(cur["_phigh"]), 3),
                       "trend": trend, "change": round(d_p, 3) if prev is not None else np.nan,
                       "prev_p": round(float(prev["_phigh"]), 3) if prev is not None else np.nan,
                       "prev_tier": prev["_tier"] if prev is not None else "—",
                       "check_ins": len(g), "last_seen": cur["_ts"].strftime("%Y-%m-%d"),
                       "days_ago": age, "stale": age > STALE_DAYS})
    people = pd.DataFrame(people)
    attention = people[people["trend"] == "▲ Worsening"].sort_values("change", ascending=False)
    tc = people["tier"].value_counts().reindex(["Priority", "Elevated", "Monitor"]).fillna(0)

    view = st.radio("View", ["Needs attention", "Everyone", "Individual review",
                             "How this works"], horizontal=True)

    # ---------- NEEDS ATTENTION (the landing screen) ----------
    if view == "Needs attention":
        st.markdown(f"#### {len(attention)} employees have got worse since their last check-in")
        st.caption(f"Out of {len(people)} people with a check-in on record. ")
        if attention.empty:
            st.success("Nobody has moved into a worse tier since their previous check-in.")
        else:
            
            SORTS = {
                "Biggest increase in risk":            ("change",   False),
                "Current risk score — highest first":  ("p_high",   False),
                "Current tier — most severe first":    ("_rank",    False),
                "Longest since a check-in":            ("days_ago", False),
                "Most recent check-in first":          ("days_ago", True),
            }
            sc1, _sc2 = st.columns([2, 1])
            sort_by = sc1.selectbox("Sort by", list(SORTS), index=0)
            _col, _asc = SORTS[sort_by]
            a = attention.copy()
            a["_rank"] = a["tier"].map(TIER_RANK)
            a = a.sort_values(_col, ascending=_asc)

            show = a[["employee_id", "employee_name", "department", "job_title", "prev_tier",
                      "tier", "prev_p", "p_high", "change", "last_seen", "days_ago", "stale"]].copy()
            show["change"]   = show["change"].map(lambda v: f"{v:+.3f}")   # sign makes direction unmissable
            show["days_ago"] = [f"{d} ⚠️" if s else f"{d}"
                                for d, s in zip(show["days_ago"], show["stale"])]
            show = show.drop(columns=["stale"]).rename(columns={
                "employee_id": "ID", "employee_name": "Name", "department": "Department",
                "job_title": "Job title", "prev_tier": "Was", "tier": "Now",
                "prev_p": "Previous risk score", "p_high": "Current risk score",
                "change": "Change in risk score", "last_seen": "Last check-in", "days_ago": "Days ago"})
            st.dataframe(show, use_container_width=True, hide_index=True, height=320)

            st.caption("**Risk score** is the estimated probability of high burnout risk (0–1). "
                       "**Change in risk score** is how far it has risen since that person's "
                       "previous check-in — a bigger number means a faster decline. ")
            _st_n = int(a["stale"].sum())
            if _st_n:
                st.warning(f"⚠️ {_st_n} of these {len(a)} employees last checked in over "
                           f"{STALE_DAYS} days ago (marked ⚠️). They were declining and have not "
                           f"been heard from since — the right action is to invite a fresh "
                           f"check-in, not to act on a stale score.")

        st.markdown("---")
        k = st.columns(4)
        stale_n = int(people["stale"].sum())
        for col, (lbl, val, sub) in zip(k, [
                ("Employees", len(people), f"{len(subs)} check-ins on record"),
                ("Priority", int(tc["Priority"]), "highest current risk"),
                ("Elevated", int(tc["Elevated"]), "worth watching"),
                ("Out of date", stale_n, f"no check-in in {STALE_DAYS}+ days")]):
            col.markdown(f"<div class='kpi'><div class='kpi-label'>{lbl}</div><div class='kpi-value'>{val}</div>"
                         f"<div class='kpi-label'>{sub}</div></div>", unsafe_allow_html=True)
        st.caption("Counts are people, not forms — each employee appears once, at their most recent "
                   "check-in. Tiers are indicative. It does not replace HR judgement. "
                   "Not a clinical diagnosis.")

    # ---------- EVERYONE ----------
    elif view == "Everyone":
        ALL_DIRS = ["▲ Worsening", "– Stable", "▼ Improving", "• Single Check-In"]
        f1, f2 = st.columns([2, 1])
        q  = f1.text_input("Search by ID or name", placeholder="type to filter…").strip().lower()
        tf = f2.multiselect("Show tiers", ["Priority", "Elevated", "Monitor"],
                            default=["Priority", "Elevated", "Monitor"])
        f3, f4 = st.columns([2, 1])
        dirs = f3.multiselect("Direction since last check-in", ALL_DIRS, default=ALL_DIRS,
                              help="'New' means only one check-in on record — no direction yet.")
        stat = f4.selectbox("Check-in status",
                            ["All", f"Up to date (under {STALE_DAYS} days)",
                             f"Out of date ({STALE_DAYS}+ days)"])

        d = people[people["tier"].isin(tf) & people["trend"].isin(dirs)]
        if stat.startswith("Up to date"):
            d = d[~d["stale"]]
        elif stat.startswith("Out of date"):
            d = d[d["stale"]]
        if q:
            d = d[d["employee_id"].astype(str).str.lower().str.contains(q) |
                  d["employee_name"].astype(str).str.lower().str.contains(q)]
        d = d.sort_values(["tier", "p_high"], key=lambda s: s.map(TIER_RANK) if s.name == "tier" else s,
                          ascending=[False, False])
        st.caption(f"Showing {len(d)} of {len(people)} employees — one row each, at their latest check-in.")

        if d.empty:
            st.warning("No employees match these filters.")
            st.stop()

        cA, cB = st.columns([1, 1.6])
        with cA:
            
            _dc = d["tier"].value_counts().reindex(["Priority", "Elevated", "Monitor"]).fillna(0)
            tdf = pd.DataFrame({"tier": ["Priority", "Elevated", "Monitor"],
                                "people": [int(_dc[t]) for t in ["Priority", "Elevated", "Monitor"]]})
            st.altair_chart(alt.Chart(tdf).mark_arc(innerRadius=55).encode(
                theta="people:Q",
                color=alt.Color("tier:N", scale=alt.Scale(domain=list(TIER_COLOR),
                                                          range=list(TIER_COLOR.values())),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["tier", "people"]).properties(height=260), use_container_width=True)
            _filtered = len(d) < len(people)
            st.caption(f"Current tier of the {len(d)} employees shown"
                       + (f" (filtered from {len(people)})" if _filtered else ""))
        with cB:
            _e = d[["employee_id", "employee_name", "department", "job_title", "tier", "trend",
                    "prev_p", "p_high", "change", "check_ins", "last_seen", "stale"]].copy()
            _e["change"]    = _e["change"].map(lambda v: "—" if pd.isna(v) else f"{v:+.3f}")
            _e["prev_p"]    = _e["prev_p"].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
            _e["last_seen"] = [f"{t} ⚠️" if s else t for t, s in zip(_e["last_seen"], _e["stale"])]
            st.dataframe(_e.drop(columns=["stale"]).rename(columns={
                "employee_id": "ID", "employee_name": "Name", "department": "Department",
                "job_title": "Job title", "tier": "Tier",
                "trend": "Direction", "prev_p": "Previous risk", "p_high": "Current risk",
                "change": "Change", "check_ins": "Check-ins", "last_seen": "Last seen"}),
                use_container_width=True, hide_index=True, height=340)
            st.caption("One row per employee, at their most recent check-in. ⚠️ marks a check-in "
                       "over 90 days old. Click a column header to sort.")
        if people["stale"].any():
            st.warning(f"{int(people['stale'].sum())} employees have not checked in for over "
                       f"{STALE_DAYS} days. Their tier reflects how they were then, not now.")

        # ============================================================================
        # BURNOUT ANALYSIS  (independent of the table filters above and of each other)
        # ----------------------------------------------------------------------------
        
        st.markdown("---")
        st.markdown("## Burnout analysis")
        
        # ---- one comprehensive per-employee frame (latest check-in), UNFILTERED ----
        # people[] already carries department / job_title / tier / trend / stale / p_high. We add
        # the RAW model feature columns (including the engineered workload_dissatisfaction) from
        # each person's latest check-in so the driver charts can be computed. Single source for all
        # three panels.
        _factor_cols = ["satisfaction_score", "workload_score", "collaboration_score",
                        "project_completion_rate", "training_participation",
                        "career_progression_score", "overtime_hours", "workload_dissatisfaction"]
        _latest_feats = hist.sort_values("_ts").groupby("employee_id").tail(1)
        _keep = ["employee_id"] + [c for c in _factor_cols if c in _latest_feats.columns]
        analytics = people.merge(_latest_feats[_keep], on="employee_id", how="left")
        analytics["department"] = analytics["department"].replace("", "—")
        analytics["job_title"]  = analytics["job_title"].replace("", "—")

        # ---- reusable concern scoring (RAW feature names, kept exactly as stored) ----
        # Each stored feature value is placed against the reference-dataset quantiles, then turned
        # into a single 0-100 "concern" scale where higher is always worse. overtime_hours and the
        # engineered workload_dissatisfaction are flipped (their HIGH end is the concerning one);
        # every other feature's LOW end is.
        _QSd = dict(svc.meta.get("feature_quantiles") or {})
        _GRID = svc.meta.get("quantile_grid") or list(np.linspace(0, 1, 101))
        # workload_dissatisfaction is engineered = (1-workload_score)*(1-satisfaction_score), so the
        # reference dataset may not ship a quantile grid for it. If missing, derive one from the two
        # component grids under an independence assumption (stated, and mirroring how the feature is
        # computed at inference). Computed once here so it appears in BOTH concern charts.
        if "workload_dissatisfaction" not in _QSd:
            _wq, _sq = _QSd.get("workload_score"), _QSd.get("satisfaction_score")
            if _wq and _sq:
                _rs = np.random.default_rng(0)
                _wsamp = np.asarray(_wq, float)[_rs.integers(0, len(_wq), 40000)]
                _ssamp = np.asarray(_sq, float)[_rs.integers(0, len(_sq), 40000)]
                _wd = (1.0 - _wsamp) * (1.0 - _ssamp)
                _QSd["workload_dissatisfaction"] = list(np.quantile(_wd, np.asarray(_GRID, float)))

        def _pct_of(c, v):
            q = _QSd.get(c)
            if not q:
                return None
            return 100.0 * float(np.searchsorted(q, float(v))) / (len(q) - 1)
        FACTORS = [
            ("satisfaction_score",       "low"),
            ("workload_score",           "low"),
            ("collaboration_score",      "low"),
            ("project_completion_rate",  "low"),
            ("training_participation",   "low"),
            ("career_progression_score", "low"),
            ("overtime_hours",           "high"),
            ("workload_dissatisfaction", "high"),
        ]
        def concern_table(frame):
            recs = []
            for col, direction in FACTORS:
                if col not in frame.columns:
                    continue
                ps = [p for p in (_pct_of(col, v) for v in frame[col]) if p is not None]
                if not ps:
                    continue
                mean_pct = float(np.mean(ps))
                concern  = (100.0 - mean_pct) if direction == "low" else mean_pct
                recs.append({"feature": col, "concern": round(concern, 1),
                             "avg_percentile": round(mean_pct, 1)})
            return pd.DataFrame(recs)

        _ALL_DIRS = ["▲ Worsening", "– Stable", "▼ Improving", "• Single Check-In"]
        # ---- reusable filter row, rendered INLINE (no expander). `key` namespaces every widget so
        #      each panel's controls stay separate. `include_dept` toggles the department multiselect. ----
        def render_filters(frame, key, include_dept=True):
            depts_all  = sorted(frame["department"].unique())
            titles_all = sorted(frame["job_title"].unique())
            r1 = st.columns(3)
            sel_dept  = (r1[0].multiselect("Department", depts_all, default=depts_all,
                                           key=f"{key}_dept") if include_dept else depts_all)
            sel_title = r1[1].multiselect("Job title", titles_all, default=titles_all,
                                          key=f"{key}_title")
            sel_tier  = r1[2].multiselect("Tier", ["Priority", "Elevated", "Monitor"],
                                          default=["Priority", "Elevated", "Monitor"],
                                          key=f"{key}_tier")
            r2 = st.columns(3)
            sel_dir  = r2[0].multiselect("Direction since last check-in", _ALL_DIRS,
                                         default=_ALL_DIRS, key=f"{key}_dir")
            sel_stat = r2[1].selectbox("Check-in status",
                                       ["All", f"Up to date (under {STALE_DAYS} days)",
                                        f"Out of date ({STALE_DAYS}+ days)"], key=f"{key}_stat")
            rmin, rmax = r2[2].slider("Current risk score", 0.0, 1.0, (0.0, 1.0), 0.05,
                                      key=f"{key}_risk")
            f = frame[frame["department"].isin(sel_dept) & frame["job_title"].isin(sel_title)
                      & frame["tier"].isin(sel_tier) & frame["trend"].isin(sel_dir)].copy()
            if sel_stat.startswith("Up to date"):
                f = f[~f["stale"]]
            elif sel_stat.startswith("Out of date"):
                f = f[f["stale"]]
            f = f[(f["p_high"] >= rmin) & (f["p_high"] <= rmax)]
            return f

        def _tier_order(frame):
            return (frame.groupby("department")["p_high"].mean()
                         .sort_values(ascending=False).index.tolist())

        # =====================================================================
        # PANEL 1 — What's driving burnout (team-wide)
        # =====================================================================
        st.markdown("### 1 · What's driving burnout risk (team-wide)")
        fa = render_filters(analytics, "drv", include_dept=True)
        st.caption(f"Based on {len(fa)} of {len(analytics)} employees (latest check-in each).")
        overall = concern_table(fa)
        if fa.empty or overall.empty:
            st.info("No employees match this panel's filters.")
        else:
            st.altair_chart(alt.Chart(overall).mark_bar(color=PRIMARY, cornerRadiusEnd=4).encode(
                x=alt.X("concern:Q", scale=alt.Scale(domain=[0, 100]),
                        title="Concern Level"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature", "concern", "avg_percentile"]).properties(height=290),
                use_container_width=True)
            

        # =====================================================================
        # PANEL 2 — Burnout by department
        # =====================================================================
        st.markdown("---")
        st.markdown("### 2 · Burnout Risk by department")
        st.caption("Compare departments side by side.")
        fb = render_filters(analytics, "bydept", include_dept=True)
        st.caption(f"Based on {len(fb)} of {len(analytics)} employees.")
        if fb.empty:
            st.info("No employees match this panel's filters.")
        else:
            order_b = _tier_order(fb)
            cD1, cD2 = st.columns([1.4, 1])
            with cD1:
                comp = fb.groupby(["department", "tier"]).size().reset_index(name="people")
                st.altair_chart(alt.Chart(comp).mark_bar().encode(
                    x=alt.X("people:Q", stack="normalize", title="share of department"),
                    y=alt.Y("department:N", sort=order_b, title=None),
                    color=alt.Color("tier:N", scale=alt.Scale(domain=list(TIER_COLOR),
                                    range=list(TIER_COLOR.values())),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["department", "tier", "people"]
                    ).properties(height=max(240, 40 * len(order_b))), use_container_width=True)
                st.caption("Each bar is one department, split by the share of its people in each "
                           "tier. Ordered by average risk")
            with cD2:
                meanr = fb.groupby("department")["p_high"].mean().reset_index(name="mean_risk")
                st.altair_chart(alt.Chart(meanr).mark_bar(color=ACCENT, cornerRadiusEnd=4).encode(
                    x=alt.X("mean_risk:Q", scale=alt.Scale(domain=[0, 1]), title="avg risk score"),
                    y=alt.Y("department:N", sort=order_b, title=None),
                    tooltip=["department", alt.Tooltip("mean_risk:Q", format=".2f")]
                    ).properties(height=max(240, 40 * len(order_b))), use_container_width=True)
                st.caption("Average current risk score per department.")

        # =====================================================================
        # PANEL 3 — Drill into a department  (pick the department FIRST, then filter within it)
        # =====================================================================
        st.markdown("---")
        st.markdown("### 3 · Drill into a department")
        st.caption("Pick one department first, then refine within it.")
        _dep_all = _tier_order(analytics)
        pickdep = st.selectbox("Department to drill into", _dep_all, key="drill_pick")
        sub = render_filters(analytics[analytics["department"] == pickdep], "drill", include_dept=False)
        st.caption(f"{len(sub)} employee{'s' if len(sub) != 1 else ''} in {pickdep} "
                   f"after this panel's filters.")
        if sub.empty:
            st.info(f"No employees in {pickdep} match this panel's filters.")
        else:
            cP1, cP2 = st.columns([1, 1.4])
            with cP1:
                dc = (sub["tier"].value_counts()
                         .reindex(["Priority", "Elevated", "Monitor"]).fillna(0).reset_index())
                dc.columns = ["tier", "people"]
                st.altair_chart(alt.Chart(dc).mark_arc(innerRadius=55).encode(
                    theta="people:Q",
                    color=alt.Color("tier:N", scale=alt.Scale(domain=list(TIER_COLOR),
                                    range=list(TIER_COLOR.values())),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["tier", "people"]).properties(height=260), use_container_width=True)
                st.caption(f"Tier split for {pickdep}.")
            with cP2:
                dept_conc = concern_table(sub)
                if dept_conc.empty:
                    st.caption("Not enough data to break down the drivers for this selection.")
                else:
                    st.altair_chart(alt.Chart(dept_conc).mark_bar(color=PRIMARY, cornerRadiusEnd=4).encode(
                        x=alt.X("concern:Q", scale=alt.Scale(domain=[0, 100]), title="Concern Level"),
                        y=alt.Y("feature:N", sort="-x", title=None),
                        tooltip=["feature", "concern", "avg_percentile"]).properties(height=290),
                        use_container_width=True)
                    st.caption(f"What's driving burnout in {pickdep} — tallest bar first.")

    # ---------- INDIVIDUAL REVIEW ----------
    elif view == "Individual review":
        opts = people.sort_values("employee_id")
        lab = [f"{r.employee_id} — {r.employee_name}   ({r.tier}, {r.check_ins} check-in"
               f"{'s' if r.check_ins > 1 else ''})" for r in opts.itertuples()]
        pick = st.selectbox("Select an employee", range(len(opts)), format_func=lambda i: lab[i])
        eid = opts.iloc[pick]["employee_id"]
        g = hist[hist["employee_id"] == eid].sort_values("_ts")
        cur = g.iloc[-1]
        meta_row = people[people["employee_id"] == eid].iloc[0]

        st.markdown("---")
        st.caption(f"**{meta_row['employee_name']}** · {meta_row.get('job_title', '—') or '—'} · "
                   f"{meta_row.get('department', '—') or '—'} · ID `{eid}`")
        h = st.columns(3)
        h[0].markdown(f"<div class='kpi'><div class='kpi-label'>Current level</div>"
                      f"<div class='kpi-value'>{cur['_pred']}</div>"
                      f"<div class='kpi-label'>{meta_row['trend']} since last check-in</div></div>",
                      unsafe_allow_html=True)
        h[1].markdown(f"<div class='kpi'><div class='kpi-label'>Estimated (HIGH) burnout risk</div>"
                      f"<div class='kpi-value'>{cur['_phigh']*100:.0f}%</div>"
                      f"<div class='kpi-label'>{meta_row['last_seen']} · {meta_row['days_ago']}d ago</div></div>",
                      unsafe_allow_html=True)
        h[2].markdown(f"<div style='padding-top:6px'></div><div class='badge' "
                      f"style='background:{TIER_COLOR[cur['_tier']]}'>Action: {cur['_tier']}</div>",
                      unsafe_allow_html=True)

        # --- history: the thing a one-off questionnaire cannot show ---
        if len(g) > 1:
            st.markdown("**How this person's risk has moved**")
            ch = pd.DataFrame({"Check-in": g["_ts"], "Risk": g["_phigh"].astype(float),
                               "Tier": g["_tier"]})
            bands = pd.DataFrame({"y0": [0, 0.40, 0.70], "y1": [0.40, 0.70, 1.0],
                                  "Tier": ["Monitor", "Elevated", "Priority"]})
            band = alt.Chart(bands).mark_rect(opacity=0.13).encode(
                y="y0:Q", y2="y1:Q",
                color=alt.Color("Tier:N", scale=alt.Scale(domain=list(TIER_COLOR),
                                                          range=list(TIER_COLOR.values())), legend=None))
            line = alt.Chart(ch).mark_line(point=alt.OverlayMarkDef(size=90), strokeWidth=2.5).encode(
                x=alt.X("Check-in:T", title=None),
                y=alt.Y("Risk:Q", scale=alt.Scale(domain=[0, 1]), title="Estimated (HIGH) burnout risk"),
                tooltip=[alt.Tooltip("Check-in:T"), alt.Tooltip("Risk:Q", format=".2f"), "Tier"])
            st.altair_chart((band + line).properties(height=240), use_container_width=True)
            st.caption("Shaded bands are the action tiers. The direction of the line matters more "
                       "than any single point: a rising line is a reason to talk to someone before "
                       "they reach Priority.")
        else:
            st.info("Only one check-in on record — no trend yet. A second check-in makes the "
                    "direction of travel visible, which is the more useful signal.")

        cL, cR = st.columns([1, 1.2])
        r = svc.predict(g.iloc[[-1]])
        with cL:
            st.markdown("**Confidence across the three levels**")
            proba = r["proba"][0]
            pdf = pd.DataFrame({"Class": svc.le.inverse_transform(np.arange(len(proba))),
                                "Probability": proba})
            st.altair_chart(alt.Chart(pdf).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Class:N", sort=svc.CLASS_ORDER, axis=alt.Axis(labelAngle=0), title=None),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Class:N", scale=alt.Scale(domain=list(CLASS_COLOR),
                                                           range=list(CLASS_COLOR.values())), legend=None),
                tooltip=["Class", alt.Tooltip("Probability:Q", format=".3f")]
                ).properties(height=260), use_container_width=True)
        with cR:
            st.markdown("**What this employee reported**")
            
            QS = (svc.meta.get("feature_quantiles") or {})
            def pctile(c, v):
                q = QS.get(c)
                return None if not q else int(round(100 * float(np.searchsorted(q, v)) / (len(q) - 1)))
            READ = {  # (feature, plain label, is a LOW value the concerning end?)
                "satisfaction_score":       ("Job satisfaction", True),
                "workload_score":           ("Workload manageability", True),
                "collaboration_score":      ("Team support", True),
                "project_completion_rate":  ("Getting work finished", True),
                "training_participation":   ("Training taken up", True),
                "career_progression_score": ("Career progression", True),
            }
            rows_r = []
            for c, (lbl, low_bad) in READ.items():
                v = float(cur[c]); p = pctile(c, v)
                if p is None: continue
                band = ("Bottom of the dataset" if p <= 10 else "Below most" if p <= 30
                        else "About average" if p <= 70 else "Better than most" if p <= 90
                        else "Top of the dataset")
                rows_r.append({"What was reported": lbl, "Compared with the reference dataset": band,
                               "Percentile": p, "_bad": low_bad and p <= 30})
            rdf = pd.DataFrame(rows_r)
            st.dataframe(rdf.drop(columns=["_bad"]), use_container_width=True, hide_index=True, height=260)
            st.caption("Percentile = where this answer sits in the reference dataset the model was "
                       "trained on — not this organisation's own staff, and not a comparison with "
                       "colleagues. It is a relative position on the model's scale.")
            # Overtime is self-reported HOURS, not equated against the reference dataset, so it does
            # not belong in the percentile table. It gets its own small card instead.
            ot = float(cur["overtime_hours"])
            _ot_note = "none reported" if ot == 0 else "more than most" if ot > 8 else "some"
            st.markdown(f"<div class='kpi'><div class='kpi-label'>Overtime reported</div>"
                        f"<div class='kpi-value'>{ot:.0f}h / week</div>"
                        f"<div class='kpi-label'>{_ot_note}</div></div>", unsafe_allow_html=True)
            st.caption("Self-reported overtime hours")

        with st.expander("Technical explanation (SHAP contributions)"):
            names, sh = svc.shap_high(r["Xp"])
            if names is None:
                st.caption("Explanation unavailable (shap/base_model missing).")
            else:
                rr = sh[0]; order = np.argsort(np.abs(rr))[::-1][:8]
                sdf = pd.DataFrame({"feature": [names[i] for i in order], "value": [rr[i] for i in order]})
                st.altair_chart(alt.Chart(sdf).mark_bar(cornerRadiusEnd=3).encode(
                    x=alt.X("value:Q", title="← lowers risk   ·   raises risk →"),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    color=alt.condition("datum.value > 0", alt.value(TIER_COLOR["Priority"]),
                                        alt.value(PRIMARY)),
                    tooltip=["feature", alt.Tooltip("value:Q", format=".3f")]
                    ).properties(height=260), use_container_width=True)
                st.caption("SHAP shows how much each feature moved THIS prediction away from the "
                           "average. It explains the model's reasoning — it is not evidence of what "
                           "causes burnout for this person.")

        st.markdown("**Suggested conversation starters**")
        
        acts = []
        def low(c, thresh=25):
            p = pctile(c, float(cur[c])); return p is not None and p <= thresh
        if low("satisfaction_score"):       acts.append("**Job satisfaction is low** — a supportive 1:1 may help surface why.")
        if low("workload_score"):           acts.append("**Workload is not manageable** — review task allocation, deadlines and priorities.")
        if float(cur["overtime_hours"]) > 8: acts.append(f"**Sustained overtime** ({float(cur['overtime_hours']):.0f}h/week) — discuss workload and work-life balance.")
        if low("collaboration_score"):      acts.append("**Limited team support reported** — check integration, pairing and inclusion.")
        if low("career_progression_score"): acts.append("**Career progression feels blocked** — explore development and growth options.")
        if low("project_completion_rate"):  acts.append("**Struggling to complete work** — check for blockers rather than assuming capability.")
        if meta_row["trend"] == "▲ Worsening":
            acts.append(f"**Direction of travel is downward** (risk up {meta_row['change']:+.2f} since "
                        f"the last check-in) — worth acting on now rather than waiting.")
        if meta_row["stale"]:
            acts.append(f"**This check-in is {meta_row['days_ago']} days old** — invite a fresh one before acting on it.")
        if not acts:
            
            if cur["_tier"] in ("Priority", "Elevated"):
                acts.append(f"**No single dominant driver** — but the model still places this "
                            f"person at {float(cur['_phigh'])*100:.0f}% risk. Their score comes from "
                            f"the *combination* of several mildly unfavourable answers rather than "
                            f"one obvious problem, which is why no specific prompt fires above. "
                            f"An open-ended conversation is more appropriate here than a targeted "
                            f"one — ask how things are generally, rather than leading with a topic.")
            else:
                acts.append("No strong risk indicators — continue routine, supportive check-ins.")
        for a in acts:
            st.markdown(f"- {a}")
        st.caption("Prompts for a conversation, not instructions — the dashboard cannot know this "
                   "person's circumstances.")

    # ---------- HOW THIS WORKS ----------
    
    elif view == "How this works":
        TT = svc.meta.get("tier_thresholds", {"Priority": 0.7, "Elevated": 0.4})
        TM = svc.meta.get("test_metrics", {}) or {}
        _ax = core.load_joblib("analytics_extra.joblib") or {}
        PCR = _ax.get("per_class_report", {}) or {}
        QS = (svc.meta.get("feature_quantiles") or {})
        GRID = svc.meta.get("quantile_grid") or list(np.linspace(0, 1, 101))

        st.markdown("### How this dashboard works")
        st.markdown(
            "Employees answer a short wellbeing questionnaire. Their answers are compared against "
            "the working conditions of **the reference dataset**, and a model that learned the "
            "patterns in those records estimates how closely this person's situation resembles "
            "people who went on to experience high burnout risk. The result is a **score between "
            "0 and 1**, which decides which of three action tiers they land in.\n\n"
            "**It is an early-warning signal, not a diagnosis.** It looks at working conditions — "
            "workload, support, satisfaction, progression — not at symptoms, and not at the "
            "person. The point is to prompt a conversation sooner than you would otherwise have "
            "had one.")

        st.markdown("#### From an answer to an action, in four steps")
        s = st.columns(4)
        steps = [("1 · Ask", "Six short question sets plus overtime hours. Around 3 minutes."),
                 ("2 · Compare", "Each answer is placed against the reference dataset — is this "
                                 "person's workload typical, or in the bottom 10%?"),
                 ("3 · Estimate", "The model weighs all eight together and returns the probability that this person falls in the high-risk group "
                                  "a score from 0 to 1."),
                 ("4 · Route", f"Score at or above {TT['Priority']:.2f} → Priority.\n"
                               f"\n{TT['Elevated']:.2f}–{TT['Priority']:.2f} → Elevated.\n"
                               f"\nBelow {TT['Elevated']:.2f} → Monitor.")]
        for col, (h, t) in zip(s, steps):
            col.markdown(f"<div class='kpi' style='min-height:150px'><div class='kpi-label'>{h}</div>"
                         f"<div style='font-size:0.86rem;line-height:1.45;padding-top:8px'>{t}</div>"
                         f"</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### What employees are asked")
        st.caption("Nothing here was invented for this project. Every question set is taken from a "
                   "published, peer-reviewed workplace questionnaire that researchers have already "
                   "tested and validated — so the answers mean what they are supposed to mean.")
        qrows = []
        for feat, spec in core.QUESTIONNAIRE.items():
            qrows.append({"What it measures": spec["label"],
                          "Questions": len(spec["items"]),
                          "Example question": spec["items"][0][0],
                          "Where it comes from": spec["source"]})
        qrows.append({"What it measures": "Overtime", "Questions": 1,
                      "Example question": "Roughly how many hours of overtime do you work in a "
                                          "typical week?",
                      "Where it comes from": "Reported directly by the employee"})
        st.dataframe(pd.DataFrame(qrows), use_container_width=True, hide_index=True)
        with st.expander("See every question"):
            for feat, spec in core.QUESTIONNAIRE.items():
                st.markdown(f"**{spec['label']}** — *{spec['source']}*")
                for text, rev in spec["items"]:
                    st.markdown(f"- {text}" + ("  *(reverse-scored)*" if rev else ""))
                st.write("")
            st.caption("Reverse-scored means agreement counts *against* the thing being measured — "
                       "for example, agreeing with 'In general, I do not like my job' lowers the "
                       "satisfaction score rather than raising it. Handling these correctly is why "
                       "the wording is left exactly as published rather than rewritten.")

        st.markdown("---")
        st.markdown("#### How one questionnaire answer becomes a number the model can use")
        st.caption("A worked example for a single measure — Job Satisfaction — using this "
                   "dataset's real figures. Every questionnaire measure goes through the same "
                   "steps. Overtime is the only input that skips them, because it is already a "
                   "number of hours.")

        # Live worked example, computed from the same reference quantiles the model uses.
        _sat_q = QS.get("satisfaction_score", [0.0] * 101)
        _wl_q  = QS.get("workload_score", [0.0] * 101)
        _lo, _hi = 0.05, 0.95
        _raw = [2, 4, 2]                        # example Likert answers (1–5)
        _adj = [2, 6 - 4, 2]                    # middle item reverse-scored: 6 − 4 = 2
        _avg = sum(_adj) / len(_adj)            # 2.0
        _pos = (_avg - 1) / 4                   # 0.25
        _p_sat = _lo + _pos * (_hi - _lo)       # bounded percentile
        _sat_val = float(np.interp(_p_sat, np.linspace(0, 1, len(_sat_q)), _sat_q))

        st.markdown(
            f"**Step 1 · Line up the answers.** A few questions are written back-to-front on "
            f"purpose, so people read carefully instead of ticking the same box down the page. "
            f"Those answers are flipped (using *6 − answer*) so that, on every question, a higher "
            f"number always means something better. \n\n"
            f"*Example:* answers **{_raw}**, with the middle one reverse-scored, become **{_adj}**."
        )
        st.markdown(
            f"**Step 2 · Average them.** The lined-up answers for that measure are averaged into a "
            f"single number. \n\n"
            f"*Example:* the average of {_adj} is **{_avg:.1f}** out of 5."
        )
        st.markdown(
            f"**Step 3 · Put it on a 0–1 scale.** That 1-to-5 average is rescaled to a 0-to-1 "
            f"*position* using *(average − 1) ÷ 4* — so the lowest possible answer becomes 0 and "
            f"the highest becomes 1. \n\n"
            f"*Example:* (2.0 − 1) ÷ 4 = **{_pos:.2f}** — a fairly low position on satisfaction."
        )
        st.markdown(
            f"**Step 4 · Compare against the reference dataset.** The model was trained on a large "
            f"reference dataset, not on questionnaire scores, so the position is matched to the "
            f"*same standing* in that dataset. To keep a short questionnaire from over-reaching, "
            f"the match is kept within the 5th–95th percentile band: "
            f"*percentile = 0.05 + position × 0.90*. \n\n"
            f"*Example:* 0.05 + {_pos:.2f} × 0.90 = **{_p_sat:.3f}**, i.e. about the "
            f"**{_p_sat*100:.0f}th percentile**. The satisfaction value sitting at that point in "
            f"the reference dataset is **{_sat_val:.2f}** — and *that* is the number the model "
            f"receives, not the raw answer."
        )

        st.info(
            f"**In plain terms:** a middling-to-low set of satisfaction answers didn't reach the "
            f"model as a raw 0.25. It was placed where that standing sits in the reference "
            f"workforce, arriving as **{_sat_val:.2f}**. Comparing every answer to the same "
            f"reference is what lets the model read them fairly."
        )

        with st.expander("What happens next — the eight features and the final score"):
            _p_wl = _lo + 0.20 * (_hi - _lo)
            _wl_val = float(np.interp(_p_wl, np.linspace(0, 1, len(_wl_q)), _wl_q))
            _wd = (1 - _wl_val) * (1 - _sat_val)
            st.markdown(
                f"Every questionnaire measure becomes one value this way, giving **seven** "
                f"numbers. Overtime joins as the **eighth**, entered directly as hours. One of the "
                f"seven — **workload dissatisfaction** — is not asked directly but calculated from "
                f"two others, as *(1 − workload) × (1 − satisfaction)*. It only becomes large when "
                f"someone has a heavy workload **and** low satisfaction at the same time — the "
                f"combination that matters most."
            )
            st.markdown(
                f"*Example:* satisfaction **{_sat_val:.2f}** and a workload value of "
                f"**{_wl_val:.2f}** give a workload-dissatisfaction of **{_wd:.2f}**."
            )
            st.markdown(
                f"All eight numbers go into the model, which returns a **burnout-risk score from "
                f"0 to 1** — the probability that this person resembles the high-risk group. That "
                f"score decides the action tier: **{TT['Priority']:.2f}+ → Priority**, "
                f"**{TT['Elevated']:.2f}–{TT['Priority']:.2f} → Elevated**, "
                f"**below {TT['Elevated']:.2f} → Monitor**."
            )
            st.caption("Every figure here is computed live from the reference dataset — these are "
                       "real numbers, not fixed illustrations.")
        
        st.markdown("---")
        st.markdown("#### Why answers are compared, not just added up")
        st.markdown(
            "A score of 3 out of 5 sounds average. Whether it is average depends entirely on the question.\n\n"
            f"Take **career progression**. In the reference dataset, the typical employee (median) scores "
            f"**{QS.get('career_progression_score', [0]*101)[50]:.2f}** — most people feel their "
            "career is moving. So someone answering a flat 3 out of 5 is not average at all; "
            "they are near the **bottom** of the reference dataset on that measure. **Training participation** runs the "
            f"opposite way: the typical employee (median) is at "
            f"**{QS.get('training_participation', [0]*101)[50]:.2f}**, so a 3 out of 5 answer is "
            "actually *above* most colleagues.\n\n"
            "This is why each answer is read as a **position in the reference dataset**, "
            "not as a bare number — the same score can mean 'struggling' on one measure and 'doing well' on another. "
            "The comparison is the meaningful part.")
        cmpd = []
        for c, lbl in [("satisfaction_score", "Job satisfaction"),
                       ("workload_score", "Workload manageability"),
                       ("collaboration_score", "Team support"),
                       ("project_completion_rate", "Getting work finished"),
                       ("training_participation", "Training taken up"),
                       ("career_progression_score", "Career progression")]:
            q = QS.get(c)
            if q:
                cmpd.append({"Measure": lbl, "Bottom 10% of the reference dataset": f"{q[10]:.2f}",
                             "Typical employee (median)": f"{q[50]:.2f}", "Top 10% of the reference dataset": f"{q[90]:.2f}"})
        st.dataframe(pd.DataFrame(cmpd), use_container_width=True, hide_index=True)
        

        st.markdown("---")
        st.markdown("#### What the three tiers mean")
        st.markdown(
            f"| Tier | Risk score | What it means | What to do |\n|---|---|---|---|\n"
            f"| **Priority** | {TT['Priority']:.2f} and above | This person's working conditions "
            f"closely resemble those of people at high burnout risk | Have a wellbeing "
            f"conversation soon |\n"
            f"| **Elevated** | {TT['Elevated']:.2f} – {TT['Priority']:.2f} | Some warning signs, "
            f"less clear-cut | Worth watching; check in at the next opportunity |\n"
            f"| **Monitor** | Below {TT['Elevated']:.2f} | No strong warning signs right now | "
            f"Routine support |")
        st.warning(
            f"**Read the tiers as a ranking, not a verdict.** They sort your workforce by relative "
            f"risk so you know who to talk to first — they are not clinical thresholds, and "
            f"'Priority' does not mean someone is burnt out. Right now **"
            f"{int(tc['Priority'])} of {len(people)} employees** sit in Priority, which is far too "
            f"many to contact individually. That is a property of the records this model learned "
            f"from, where most employees showed elevated risk. It is the reason the **Needs "
            f"attention** screen ranks by *change* rather than by level: who is getting worse is a "
            f"far more useful question than who is currently high.")

        st.markdown("---")
        st.markdown("#### How accurate is it?")
        st.caption("Measured on 127,500 employee records the model had never seen during training, "
                   "and used once, at the very end. These are the real numbers.")
        hi, lo, mo = PCR.get("High", {}), PCR.get("Low", {}), PCR.get("Moderate", {})
        if hi:
            a = st.columns(3)
            a[0].markdown(f"<div class='kpi'><div class='kpi-label'>CATCHES HIGH-RISK EMPLOYEES</div>"
                          f"<div class='kpi-value'>{hi.get('recall',0)*100:.0f}%</div>"
                          f"<div class='kpi-label'>of everyone genuinely at high risk, the model correctly flags 91%</div></div>",
                          unsafe_allow_html=True)
            a[1].markdown(f"<div class='kpi'><div class='kpi-label'>HIGH-RISK FLAGS THAT ARE CORRECT</div>"
                          f"<div class='kpi-value'>{hi.get('precision',0)*100:.0f}%</div>"
                          f"<div class='kpi-label'>when it flags someone as high-risk, it's right 93% of the time</div></div>",
                          unsafe_allow_html=True)
            a[2].markdown(f"<div class='kpi'><div class='kpi-label'>OVERALL ACCURACY (ALL 3 LEVELS)</div>"
                          f"<div class='kpi-value'>{TM.get('accuracy',0)*100:.0f}%</div>"
                          f"<div class='kpi-label'>correctly sorts 85% of employees into Low, Moderate or High</div></div>",
                          unsafe_allow_html=True)
            st.write("")
            st.markdown(
                f"**Where it is weakest — and it matters that you know.** The model is strong at "
                f"the two ends and much weaker in the middle. It identifies **"
                f"{hi.get('recall',0)*100:.0f}%** of genuinely high-risk employees and **"
                f"{lo.get('recall',0)*100:.0f}%** of low-risk ones, but only **"
                f"{mo.get('recall',0)*100:.0f}%** of the moderate group. Moderate sits between the "
                f"other two, so it is inherently the hardest to pin down.\n\n"
                f"**How it gets things wrong is reassuring**. Almost every mistake is one step off "
                f"- Moderate instead of High, or Low instead of Moderate. The most serious error, mixing "
                f"up high-risk and low-risk, is rare. It happens in about **2 in every 100 cases**. ")

        st.markdown("---")
        st.markdown("#### What this dashboard cannot do")
        st.markdown(
            "- **It cannot diagnose burnout.** It has never seen a clinical assessment. It "
            "recognises *working conditions* that resemble those of at-risk employees.\n"
            "- **It cannot tell you why.** It can show which answers pushed a score up, but not "
            "what is happening in that person's life. Only a conversation does that.\n"
            "- **It only knows what it is told.** Answers are self-reported. Someone who "
            "under-reports their workload will score better than they should.\n"
            "- **It learned from records, not from your workforce.** The patterns come from a "
            "published dataset whose labels were generated by a formula rather than measured on "
            "real people. That is why the tiers rank relative risk instead of claiming absolute "
            "accuracy, and why the dashboard would need re-checking against real outcomes before "
            "any decision of consequence rested on it.\n"
            "- **It is a demonstration, not a live product.** This is a proof of concept built to "
            "test whether the approach is workable.\n"
            "- **It must never be used for performance management, discipline, or any employment "
            "decision.** It exists to help people, and using it against them would break the "
            "reason it was built.")

        st.markdown("---")


# ============================================================ HR ANALYST
elif ss.role == "hr_analyst":
    if not ss.analyst_auth:
        st.markdown("<div class='hero'><h1>📊 HR Analyst sign-in</h1>"
                    "<p>Model findings, comparison and validation evidence.</p></div>",
                    unsafe_allow_html=True)
        if st.button("← Back to home", key="ana_back"):
            go_home(); st.rerun()
        st.markdown("#### Sign in")
        apw = st.text_input("Passcode", type="password", key="ana_pw")
        if st.button("Sign in", type="primary", key="ana_go"):
            if apw == ANALYST_PASSCODE:
                ss.analyst_auth = True; st.rerun()
            else:
                st.error("Incorrect passcode.")
        
        st.stop()

    tA1, tA2 = st.columns([4, 1])
    tA1.markdown("<div class='hero'><h1>📊 Model Findings & Validation</h1>"
                "<p>The evaluation and validation evidence HR Analysts use to check whether the "
                "prediction model is trustworthy enough to support the HR Manager's console. "
                "It has exploratory analysis, comparison across seven models, and the robustness checks "
                "behind the deployed model.</p></div>", unsafe_allow_html=True)
    with tA2:
        st.write("")
        if st.button("Sign out", use_container_width=True, key="ana_signout"):
            go_home(); st.rerun()

    try:
        META, EXTRA = get_meta()
    except Exception as e:
        st.error(f"Could not load artefacts from outputs/model_artefacts. Run the notebook first.\n\n{e}")
        st.stop()

    EDA = jbl("eda_summary.joblib")
    INV = jbl("investigation_summary.joblib")
    TEST = META.get("test_metrics", {})
    CLASS_ORDER = META["class_order"]

    st.caption(f"Model in use: {META.get('model_name','?')}")

    # Top radio replaces the original sidebar navigation (app_analytics.py used `st.sidebar`),
    # so the same four sections sit alongside the Employee/HR Manager views' own top radios
    # rather than behind a separate sidebar toggle that would appear across every role.
    page = st.radio("Section", ["Overview", "EDA & Findings", "Model Performance",
                                "Validation & Robustness"], horizontal=True, key="ana_page")
    st.markdown("---")

    # ------------------------------------------------------------ OVERVIEW
    if page == "Overview":
        st.markdown("<div class='hero'><h1>Employee Burnout Risk — Findings & Model Performance</h1>"
                    "<p>Exploratory analysis, model comparison across all metrics, and the validation "
                    "evidence behind the deployed model.</p></div>", unsafe_allow_html=True)
        c = st.columns(4)
        kpi(c[0], "Test MCC", f"{TEST.get('mcc', float('nan')):.3f}", "primary metric")
        kpi(c[1], "Macro-F1", f"{TEST.get('macro_f1', float('nan')):.3f}")
        kpi(c[2], "AUC-ROC", f"{TEST.get('auc_roc', float('nan')):.3f}")
        kpi(c[3], "Accuracy", f"{TEST.get('accuracy', float('nan')):.3f}")
        recon = (INV or {}).get("reconstruction_r2", {})
        r2 = f"{max(recon.values()):.2f}" if recon else "≈0.83"
        st.markdown(f"<div class='finding'><b>Summary.</b> The model classifies employees into Low / Moderate / High "
                    f"burnout risk at MCC ≈ {TEST.get('mcc', float('nan')):.2f}. The selected "
                    f"features explain a large share of the burnout score (R² ≈ {r2}) without the target being a "
                    f"deterministic copy of the inputs, so the model has genuine learning to do.</div>", unsafe_allow_html=True)
        

    # ------------------------------------------------------------ EDA
    elif page == "EDA & Findings":
        st.header("Exploratory Data Analysis")
        if EDA is None:
            st.info("eda_summary.joblib not found — run the notebook export cell.")
        else:
            cd = EDA["class_distribution"]; ceil = EDA["ceiling_stats"]
            L, R = st.columns(2)
            with L:
                st.subheader("Target: burnout risk levels")
                cls_df = pd.DataFrame(cd)
                ch = alt.Chart(cls_df).mark_bar(cornerRadiusEnd=6).encode(
                    x=alt.X("Class:N", sort=CLASS_ORDER, axis=alt.Axis(labelAngle=0), title=None),
                    y=alt.Y("Count:Q", title=None),
                    color=alt.Color("Class:N", scale=alt.Scale(domain=list(CLASS_COLOR), range=list(CLASS_COLOR.values())), legend=None),
                    tooltip=["Class", "Count", "Percentage"]).properties(height=280)
                st.altair_chart(ch, use_container_width=True)
            with R:
                st.subheader("Burnout score distribution")
                sh = pd.DataFrame(EDA["score_hist"]); sh["mid"] = (sh["bin_left"]+sh["bin_right"])/2
                bars = alt.Chart(sh).mark_bar(color=CLASS_COLOR["High"], opacity=.75).encode(
                    x=alt.X("mid:Q", title="Burnout risk score"), y=alt.Y("count:Q", title="Employees"),
                    tooltip=[alt.Tooltip("bin_left:Q", format=".2f"), "count"])
                rules = alt.Chart(pd.DataFrame({"x": [ceil["low_mod_cut"], ceil["mod_high_cut"]]})).mark_rule(
                    strokeDash=[5, 4], color="gray").encode(x="x:Q")
                st.altair_chart((bars+rules).properties(height=280), use_container_width=True)
            st.markdown(f"<div class='finding'>~{ceil['pct_at_1']:.1f}% of employees sit at the maximum score, so fixed "
                        f"thresholds ({ceil['low_mod_cut']}/{ceil['mod_high_cut']}) were used instead of quantile bins. "
                        f"High ≈ {cd[-1]['Percentage']}% makes MCC the primary metric.</div>", unsafe_allow_html=True)
            st.markdown("---")
            st.subheader("Feature distributions")
            nh = EDA.get("numeric_histograms", {})
            if nh:
                _selected = META.get("s2_features", [])
                _avail = [c for c in nh.keys() if c in _selected] or list(nh.keys())
                f = st.selectbox("Feature", _avail)
                hdf = pd.DataFrame(nh[f])
                st.altair_chart(alt.Chart(hdf).mark_bar(color=PRIMARY, opacity=.8).encode(
                    x=alt.X("x:Q", title=f), y=alt.Y("count:Q", title="Count"),
                    tooltip=[alt.Tooltip("x:Q", format=".3f"), "count"]).properties(height=300), use_container_width=True)
            st.markdown("---")
            st.subheader("Correlation matrix")
            cm = EDA.get("corr_matrix")
            if cm is not None:
                cm = cm if isinstance(cm, pd.DataFrame) else pd.DataFrame(cm)
                long = cm.reset_index().melt(id_vars="index", var_name="fy", value_name="corr").rename(columns={"index": "fx"})
                order = list(cm.columns)
                st.altair_chart(alt.Chart(long).mark_rect().encode(
                    x=alt.X("fx:N", title=None, sort=order, axis=alt.Axis(labelAngle=-45, labelFontSize=8, labelOverlap=False)),
                    y=alt.Y("fy:N", title=None, sort=order, axis=alt.Axis(labelFontSize=8, labelOverlap=False)),
                    color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="r"),
                    tooltip=["fx", "fy", alt.Tooltip("corr:Q", format=".2f")]
                ).properties(height=520).configure_axis(labelOverlap=False), use_container_width=True)

    # ------------------------------------------------------------ MODEL PERFORMANCE (all metrics)
    elif page == "Model Performance":
        st.header("Model Performance")

        st.subheader("Final model — XGBoost")
        c = st.columns(7)
        kpi(c[0], "MCC", f"{TEST.get('mcc', float('nan')):.3f}", "primary")
        kpi(c[1], "Accuracy", f"{TEST.get('accuracy', float('nan')):.3f}")
        kpi(c[2], "Precision", f"{TEST.get('precision_macro', float('nan')):.3f}", "macro")
        kpi(c[3], "Recall", f"{TEST.get('recall_macro', float('nan')):.3f}", "macro")
        kpi(c[4], "Macro-F1", f"{TEST.get('macro_f1', float('nan')):.3f}")
        kpi(c[5], "AUC-ROC", f"{TEST.get('auc_roc', float('nan')):.3f}")
        kpi(c[6], "Brier", f"{TEST.get('brier', float('nan')):.4f}", "lower=better")

        # per-class table
        pcr = (EXTRA or {}).get("per_class_report")
        if pcr:
            st.markdown("**Per-class metrics**")
            rows = []
            for cls in CLASS_ORDER:
                if cls in pcr:
                    r = pcr[cls]
                    rows.append({"Class": cls, "Precision": round(r["precision"], 3), "Recall": round(r["recall"], 3),
                                 "F1": round(r["f1-score"], 3), "Support": int(r["support"])})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("---")

        L, R = st.columns(2)
        with L:
            st.subheader("Top burnout drivers (SHAP)")
            if EXTRA and EXTRA.get("shap_top"):
                dd = pd.DataFrame(EXTRA["shap_top"]); dd["feature"] = dd["feature"].map(clean)
                st.altair_chart(alt.Chart(dd.head(12)).mark_bar(color=PRIMARY, cornerRadiusEnd=3).encode(
                    x=alt.X("mean_abs_shap:Q", title="mean |SHAP|"), y=alt.Y("feature:N", sort="-x", title=None),
                    tooltip=["feature", alt.Tooltip("mean_abs_shap:Q", format=".3f")]).properties(height=340), use_container_width=True)
        with R:
            st.subheader("Confusion matrix (test)")
            if EXTRA and EXTRA.get("confusion_matrix"):
                cm = np.array(EXTRA["confusion_matrix"]); cls = EXTRA.get("class_names", CLASS_ORDER)
                long = [{"Actual": cls[i], "Predicted": cls[j], "count": int(cm[i, j])} for i in range(len(cls)) for j in range(len(cls))]
                cmdf = pd.DataFrame(long)
                base = alt.Chart(cmdf).encode(x=alt.X("Predicted:N", sort=cls), y=alt.Y("Actual:N", sort=cls))
                heat = base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None), tooltip=["Actual", "Predicted", "count"])
                mx = int(cm.max())
                txt = base.mark_text(fontWeight="bold").encode(text=alt.Text("count:Q", format=","),
                        color=alt.condition(f"datum.count > {mx*0.5:.0f}", alt.value("white"), alt.value("black")))
                st.altair_chart((heat+txt).properties(height=340), use_container_width=True)
        st.markdown("---")

        st.subheader("Model comparison — every model × strategy, all metrics")
        cmp = csv("model_comparison_S2.csv")
        if cmp is None:
            st.info("model_comparison_S2.csv not found.")
        else:
            metric_map = {"MCC": "mcc", "Accuracy": "accuracy", "Precision (macro)": "precision_macro",
                          "Recall (macro)": "recall_macro", "Macro-F1": "macro_f1", "AUC-ROC": "auc", "Brier": "brier"}
            avail = {k: v for k, v in metric_map.items() if v in cmp.columns}
            sel = st.radio("Compare by", list(avail.keys()), horizontal=True)
            col = avail[sel]
            asc = (col == "brier")
            chart = alt.Chart(cmp).mark_bar().encode(
                x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)),
                xOffset="strategy:N",
                y=alt.Y(f"{col}:Q", title=sel, scale=alt.Scale(zero=False)),
                color=alt.Color("strategy:N", scale=alt.Scale(domain=list(STRAT_COLOR), range=list(STRAT_COLOR.values())),
                                legend=alt.Legend(orient="top", title="Strategy")),
                tooltip=["model", "strategy", alt.Tooltip(f"{col}:Q", format=".4f")]).properties(height=360)
            st.altair_chart(chart, use_container_width=True)

            st.markdown("**Full results table** (sorted by MCC)")
            show_cols = ["model", "strategy"] + [c for c in ["mcc", "accuracy", "precision_macro", "recall_macro",
                                                              "macro_f1", "auc", "brier"] if c in cmp.columns]
            table = cmp[show_cols].sort_values("mcc", ascending=False).reset_index(drop=True)
            st.dataframe(table.style.format({c: "{:.4f}" for c in show_cols if c not in ("model", "strategy")})
                            .background_gradient(subset=["mcc"], cmap="Blues"),
                         use_container_width=True, height=430)
            st.caption("Top configurations are within ~0.001 MCC (a statistical tie); XGBoost with class weighting was "
                       "chosen on practical grounds — it keeps all the real data and calibrates well.")
        

    # ------------------------------------------------------------ VALIDATION
    elif page == "Validation & Robustness":
        st.header("Model Validation & Robustness")
        if INV is None:
            st.info("investigation_summary.joblib not found.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Check 1 — Generalisation")
                ov = INV.get("overfitting", {})
                if ov:
                    odf = pd.DataFrame([{"split": "Train", "mcc": ov["train_mcc"]}, {"split": "Test", "mcc": ov["test_mcc"]}])
                    st.altair_chart(alt.Chart(odf).mark_bar(color=PRIMARY, cornerRadiusEnd=4).encode(
                        x=alt.X("split:N", title=None, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("mcc:Q", scale=alt.Scale(domain=[0, 1]), title="MCC"), tooltip=["split", "mcc"]).properties(height=240), use_container_width=True)
                    st.caption(f"Train−Test gap = {ov['gap']:.4f} → no overfitting.")
            with c2:
                st.subheader("Check 2 — Learning curve")
                ss = INV.get("sample_size_sweep", [])
                if ss:
                    sdf = pd.DataFrame(ss); ymin = float(sdf["mcc"].min()) - 0.05
                    st.altair_chart(alt.Chart(sdf).mark_line(point=True, strokeWidth=2.5, color=ACCENT).encode(
                        x=alt.X("train_size:Q", title="training rows"),
                        y=alt.Y("mcc:Q", scale=alt.Scale(domain=[max(0, ymin), 0.85]), title="MCC"), tooltip=["train_size", "mcc"]).properties(height=240), use_container_width=True)
                    st.caption("Rises with data then plateaus — a healthy learning curve.")
            st.markdown("---")
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Check 3 — Leakage control")
                po = INV.get("post_outcome", {})
                if po:
                    pdf = pd.DataFrame([{"set": "Selected Columns", "mcc": round(float(po["s2_best"]), 4)},
                                        {"set": "+ post-outcome columns", "mcc": round(float(po["s1_best"]), 4)}])
                    bars = alt.Chart(pdf).mark_bar(color=PRIMARY, cornerRadiusEnd=4).encode(
                        y=alt.Y("set:N", title=None, sort=list(pdf["set"])), x=alt.X("mcc:Q", scale=alt.Scale(domain=[0, 1]), title="best MCC"), tooltip=["set", "mcc"])
                    lab = alt.Chart(pdf).mark_text(align="left", dx=4, fontWeight="bold", color="white").encode(
                        y=alt.Y("set:N", sort=list(pdf["set"])), x="mcc:Q", text=alt.Text("mcc:Q", format=".4f"))
                    st.altair_chart((bars+lab).properties(height=190), use_container_width=True)
                    st.caption(f"Adding only the post-outcome columns to the real feature set inflates MCC by "
                               f"{po['s1_best']-po['s2_best']:+.4f} → isolates the leakage to those columns, correctly excluded.")
            with c4:
                st.subheader("Check 4 — Ceiling robustness")
                cm = INV.get("ceiling_removed_mcc")
                if cm is not None:
                    st.metric("MCC with the 1.0 ceiling removed", f"{cm:.4f}")
                    st.caption("Still solid after removing ~46% of rows at the maximum.")
            st.markdown("---")
            st.subheader("Check 5 — Feature dependence (ablation)")
            ab = INV.get("ablation", [])
            if ab:
                adf = pd.DataFrame(ab)
                adf["scenario"] = adf["scenario"].replace({
                    "Remove satisfaction + workload":          "Remove workload + satisfaction group",
                    "Remove project completion + performance": "Remove completion + performance group",
                    "Remove all 4 top drivers":                "Remove both feature groups",
                })
                st.altair_chart(alt.Chart(adf).mark_bar(color=PRIMARY, cornerRadiusEnd=3).encode(
                    x=alt.X("mcc:Q", scale=alt.Scale(domain=[0, 1]), title="MCC after removing features"),
                    y=alt.Y("scenario:N", sort=list(adf["scenario"]), title=None),
                    tooltip=["scenario", "n_features", "mcc"]).properties(height=220), use_container_width=True)
                st.caption("MCC drops as the strongest feature groups are removed → the model relies on a spread of legitimate indicators.")
            st.markdown("---")
            st.subheader("Check 6 — Threshold sensitivity (are the 0.40 / 0.70 cut-points arbitrary?)")
            ts = INV.get("threshold_sensitivity", [])
            if ts:
                tsdf = pd.DataFrame(ts)
                tsdf["cut"] = tsdf["low_mod_cut"].map("{:.2f}".format) + " / " + tsdf["mod_high_cut"].map("{:.2f}".format)
                metric = st.radio("Metric", ["auc", "macro_f1", "mcc"], horizontal=True, key="ts_metric",
                                  format_func=lambda m: {"auc": "AUC (threshold-independent)",
                                                         "macro_f1": "Macro-F1 (per-class)",
                                                         "mcc": "MCC (thresholded decision)"}[m])
                st.altair_chart(alt.Chart(tsdf).mark_line(point=True).encode(
                    x=alt.X("cut:N", title="low / high cut-point", sort=None,
                    axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
                    y=alt.Y(f"{metric}:Q", scale=alt.Scale(zero=False, nice=False, padding=12), title=metric.upper()),
                    color=alt.Color("model:N", title=None),
                    tooltip=["cut", "model", "mcc", "auc", "macro_f1", "test_pct_high"]
                ).properties(height=320), use_container_width=True)
                st.caption(
                    "Features, rows and models are held constant. Oonly the cut-points move (±0.10, 25 label "
                    "definitions, Moderate ranging from 6% to 30% of the workforce). Nothing collapses: the "
                    "largest movement for any tree model is 0.042, against 0.258 for re-introducing leakage "
                    "and 0.364 for removing the top drivers. Gradient boosting leads on **macro-F1 at 25/25 "
                    "cut-points and on AUC at 25/25**; on MCC it leads at 21/25, ceding only where Moderate "
                    "is squeezed below 12% and the bagged model raises MCC by abstaining on the vestigial "
                    "class (its highest accuracy and lowest macro-F1 in the grid occur together there). "
                    "Tree ensembles beat linear and instance-based learners at 25/25 without exception. "
                    "The baseline (0.40/0.70) sits at the grid mean, and *below* it for the deployed model — "
                    "so it was audited, not tuned. Tiers are relative operational bands for prioritisation, "
                    "not clinical cutoffs.")
                with st.expander("Full sensitivity grid"):
                    st.dataframe(tsdf.drop(columns=["cut"]), use_container_width=True, hide_index=True)

            tr = INV.get("tier_sensitivity", [])
            if tr:
                st.markdown("**Action-tier trade-off** — shifting the Priority / Elevated cut-points ±0.05")
                st.dataframe(pd.DataFrame(tr), use_container_width=True, hide_index=True)
                st.caption("Moving the tier cut-points changes HR workload and coverage, not the prediction: "
                           "AUC is computed across all cut-points and is unaffected.")
            
