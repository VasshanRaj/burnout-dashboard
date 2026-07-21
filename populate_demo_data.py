"""
populate_demo_data.py — run from the PROJECT ROOT (FYP Code/):
    python populate_demo_data.py --reset                    # wipe everything, rebuild with accounts
    python populate_demo_data.py --reset --password mypw    # choose the shared demo password
    python populate_demo_data.py --reset --employees 120    # more people

Builds a complete demo dataset: employees (now WITH department + job title), their check-in
history, AND their login accounts.

IMPORTANT — how this works and why it is honest:
  Every record is produced by simulating QUESTIONNAIRE ANSWERS (raw 1-5 Likert clicks) and then
  running them through the REAL scoring key and the REAL calibrated model, exactly as if a person
  had filled the form in. No prediction, probability or tier is fabricated: they are all genuine
  model output. Only the ANSWERS are simulated.

  Each employee has a latent wellbeing level that drifts across their check-ins, so the data
  contains improving, stable and deteriorating trajectories - which is what makes the dashboard's
  trend view meaningful rather than decorative.

  DEPARTMENT & JOB TITLE (new): every employee is assigned one of core.DEPARTMENTS and one of
  core.JOB_TITLES - the SAME canonical lists the sign-up dropdown uses, so the demo can never
  produce a value the app would not. Departments are given slightly different baseline wellbeing
  so the HR Manager's new "burnout by department" charts show real variation rather than six
  identical bars. This is a demonstration convenience (we already simulate every answer); it does
  not touch the model, the scoring key, or any real person.

  Accounts are created AFTER the check-ins, via the same register_employee() path a real employee
  uses - now passing department + job title, which is written on the "claim existing ID" branch.
  This keeps first_seen pointing at each person's genuine first check-in rather than at setup time.

  Passwords are hashed with PBKDF2 (200k rounds, per-user salt) and never stored in readable form.
  A single shared demo password is a demonstration convenience, not a security design.
"""
import sys, os, argparse
from datetime import datetime, timedelta

sys.path.insert(0, "burnout_system")
import numpy as np
import pandas as pd
import burnout_core as core

# For a "good" employee, is a HIGH raw response good (+1) or bad (-1)?
# workload is -1: a thriving employee reports LOW demand ("I have too much work to do" -> 1).
# The demand -> health inversion then happens inside score_features, as it does for a real user.
GOOD_DIRECTION = {
    "satisfaction_score": +1,
    "workload_score": -1,
    "collaboration_score": +1,
    "project_completion_rate": +1,
    "training_participation": +1,
    "career_progression_score": +1,
}

FIRST = ["Aisha", "Marcus", "Priya", "Daniel", "Mei", "Kwame", "Sofia", "Arjun", "Hannah", "Tomas",
         "Leila", "Ryan", "Ngozi", "Elena", "Hiroshi", "Farah", "Oliver", "Ananya", "Diego", "Grace",
         "Yusuf", "Clara", "Ravi", "Nadia", "Sean", "Ines", "Bao", "Maya", "Lucas", "Zara",
         "Ethan", "Rina", "Omar", "Chloe", "Viktor", "Amara", "Jonas", "Lin", "Talia", "Andre",
         "Keiko", "Samir", "Nora", "Felix", "Imani", "Dmitri", "Aria", "Tobias", "Sana", "Idris"]
LAST = ["Rahman", "Silva", "Nair", "O'Brien", "Chen", "Mensah", "Rossi", "Kapoor", "Weber", "Novak",
        "Haddad", "Murphy", "Okafor", "Petrova", "Tanaka", "Aziz", "Bennett", "Reddy", "Morales",
        "Adeyemi", "Karim", "Fischer", "Iyer", "Hassan", "Doyle", "Costa", "Tran", "Sharma",
        "Almeida", "Khan"]

# ---------------------------------------------------------------- organisational shape
# Departments and titles come straight from burnout_core so they always match the app's dropdown.
# DEPT_WEIGHTS spreads people across every department with none left empty (good donuts). Bigger
# functions (Engineering, Sales & Marketing, Customer Support) get more headcount; specialist
# functions (Law & Compliance, Administration) get less - but still enough to chart.
DEPT_WEIGHTS = {
    "Engineering":              0.14,
    "IT":                       0.10,
    "Research & Development":    0.08,
    "Sales & Marketing":        0.15,
    "Customer Support":         0.12,
    "Finance & Accounting":     0.09,
    "Human Resources":          0.06,
    "Operations":               0.10,
    "Logistics & Supply Chain": 0.07,
    "Administration":           0.05,
    "Law & Compliance":         0.04,
}
TITLE_WEIGHTS = [0.42, 0.22, 0.14, 0.12, 0.06, 0.04]   # aligns to core.JOB_TITLES order (IC most common)

# Baseline wellbeing centre per department (0 = struggling, 1 = thriving). Front-line, high-tempo
# and target-driven functions sit lower; autonomous / support functions sit higher. This spread is
# what makes the "burnout by department" charts vary rather than showing identical bars.
DEPT_WELLBEING = {
    "Engineering":              0.55,
    "IT":                       0.52,
    "Research & Development":    0.60,
    "Sales & Marketing":        0.42,
    "Customer Support":         0.40,
    "Finance & Accounting":     0.53,
    "Human Resources":          0.57,
    "Operations":               0.48,
    "Logistics & Supply Chain": 0.45,
    "Administration":           0.58,
    "Law & Compliance":         0.50,
}


def simulate_answers(rng, wellbeing):
    """Latent wellbeing (0=struggling, 1=thriving) -> raw 1-5 Likert clicks + overtime hours."""
    responses = {}
    for feat, spec in core.QUESTIONNAIRE.items():
        direction = GOOD_DIRECTION.get(feat, +1)
        target = wellbeing if direction > 0 else (1.0 - wellbeing)
        # per-scale wobble: people are not perfectly consistent across constructs
        target = float(np.clip(target + rng.normal(0, 0.12), 0.02, 0.98))
        clicks = []
        for _text, rev in spec["items"]:
            adj = 1.0 + 4.0 * target + rng.normal(0, 0.45)   # intended (reverse-corrected) response
            raw = (6.0 - adj) if rev else adj                # what the person actually clicks
            clicks.append(int(np.clip(round(raw), 1, 5)))
        responses[feat] = clicks

    # Overtime: the dataset's median is 0 (>50% do none), so model it as a hurdle -
    # probability of ANY overtime rises as wellbeing falls, then a magnitude on top.
    p_any = 0.12 + 0.60 * (1.0 - wellbeing)
    overtime = 0.0
    if rng.random() < p_any:
        overtime = float(np.clip(round(rng.gamma(2.0, 3.0 + 6.0 * (1.0 - wellbeing)), 1), 0.5, 45))
    return responses, overtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="wipe ALL check-ins and accounts first")
    ap.add_argument("--employees", type=int, default=200)
    ap.add_argument("--password", type=str, default="demo1234",
                    help="shared demo password for every generated account")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    svc = core.BurnoutService()

    if not hasattr(core, "register_employee"):
        print("\n  *** burnout_core.py has no register_employee(). Apply the employee-auth")
        print("      changes to burnout_core.py before running this. ***\n")
        return
    if not hasattr(core, "DEPARTMENTS") or not hasattr(core, "JOB_TITLES"):
        print("\n  *** burnout_core.py has no DEPARTMENTS / JOB_TITLES lists. Apply the")
        print("      department + job-title changes to burnout_core.py before running this. ***\n")
        return

    if args.reset:
        core.clear_submissions()
        print("Wiped: all check-ins and all accounts.\n")

    print(f"Model version for these records: {core.model_version()}")
    if "/s" not in core.model_version():
        print("\n  *** WARNING: model_version has no scoring-version suffix. The SCORING_VERSION")
        print("      patch has not been applied. Fix burnout_core.py first, or every row written")
        print("      now will be untraceable to the scoring key that produced it. ***\n")
        return

    used, rows, people = set(), [], []
    now = datetime.now()

    # Department sampling probabilities, aligned to core.DEPARTMENTS order and normalised so they
    # always sum to 1 even if the weights above are edited. Any department missing from
    # DEPT_WEIGHTS falls back to the average weight, so no department is ever unreachable.
    _avg_w = 1.0 / len(core.DEPARTMENTS)
    _dept_w = np.array([DEPT_WEIGHTS.get(dpt, _avg_w) for dpt in core.DEPARTMENTS], dtype=float)
    _dept_p = _dept_w / _dept_w.sum()

    # ---------------------------------------------------------------- 1. check-ins
    for i in range(args.employees):
        while True:
            name = f"{FIRST[rng.integers(len(FIRST))]} {LAST[rng.integers(len(LAST))]}"
            if name not in used:
                used.add(name); break
        eid = f"E{1001 + i}"

        department = str(rng.choice(core.DEPARTMENTS, p=_dept_p))
        job_title  = str(rng.choice(core.JOB_TITLES,  p=TITLE_WEIGHTS))
        people.append((eid, name, department, job_title))

        # Wellbeing is centred on the employee's department, with a small seniority lift so that
        # more senior staff skew slightly healthier - then clipped to a sensible range.
        centre = DEPT_WELLBEING.get(department, 0.50)
        seniority = core.JOB_TITLES.index(job_title) / (len(core.JOB_TITLES) - 1)  # 0..1
        centre = centre + 0.06 * (seniority - 0.5)
        wellbeing = float(np.clip(rng.normal(centre, 0.15), 0.03, 0.97))

        n_checkins = int(rng.choice([1, 2, 3, 4], p=[0.22, 0.33, 0.28, 0.17]))
        drift = float(rng.choice([-0.13, -0.05, 0.0, 0.05, 0.13], p=[0.18, 0.22, 0.20, 0.22, 0.18]))

        days = sorted(rng.integers(3, 180, size=n_checkins).tolist(), reverse=True)
        for k, dday in enumerate(days):
            w = float(np.clip(wellbeing + drift * k + rng.normal(0, 0.05), 0.02, 0.98))
            responses, overtime = simulate_answers(rng, w)

            feats = svc.features_for(responses, overtime)
            res = svc.predict(pd.DataFrame([feats]))
            ts = (now - timedelta(days=int(dday), hours=int(rng.integers(0, 9)),
                                  minutes=int(rng.integers(0, 60)))).strftime("%Y-%m-%d %H:%M:%S")

            record = {"employee_id": eid, "employee_name": name, "timestamp": ts, **feats,
                      "predicted_class": res["pred"][0],
                      "prob_high": round(float(res["p_high"][0]), 4),
                      "risk_tier": res["tiers"][0]}
            core.save_submission(record)
            rows.append({"employee_id": eid, "name": name, "department": department,
                         "job_title": job_title, "checkin": k + 1,
                         "wellbeing": round(w, 2), "overtime": overtime,
                         "p_high": record["prob_high"], "tier": record["risk_tier"]})

    df = pd.DataFrame(rows)
    print(f"Inserted {len(df)} check-ins for {df.employee_id.nunique()} employees.")

    # ---------------------------------------------------------------- 2. accounts
    # Registered AFTER the check-ins, through the SAME path a real employee uses - now passing
    # department + job title. This exercises the "claim an existing employee_id" branch (which
    # writes the department/title with COALESCE so nothing is clobbered) and keeps first_seen on
    # the true first check-in.
    made, failed = 0, []
    for eid, name, department, job_title in people:
        ok, msg = core.register_employee(eid, name, args.password, department, job_title)
        if ok:
            made += 1
        else:
            failed.append((eid, msg))
    print(f"Created {made} accounts (PBKDF2, per-user salt - no plaintext stored).")
    for eid, msg in failed:
        print(f"  ! {eid}: {msg}")

    ok_login, _ = core.authenticate_employee(people[0][0], args.password)
    bad_login, _ = core.authenticate_employee(people[0][0], args.password + "x")
    bad_id, _ = core.authenticate_employee("E9999", args.password)
    print(f"  Login check - correct password accepted : {'PASS' if ok_login else 'FAIL'}")
    print(f"  Login check - wrong password rejected   : {'PASS' if not bad_login else 'FAIL'}")
    print(f"  Login check - unknown ID rejected       : {'PASS' if not bad_id else 'FAIL'}")

    # ---------------------------------------------------------------- 3. report
    print(f"\n{'='*72}")
    print("DEMO CREDENTIALS - every employee shares one password for demonstration only:")
    print(f"  password : {args.password}")
    print(f"  IDs      : {people[0][0]} .. {people[-1][0]}")
    print(f"  example  : {people[0][0]} / {args.password}   ({people[0][1]})")
    print(f"  HR Manager passcode : see HR_MANAGER_PASSCODE in app.py")
    print(f"  HR Analyst passcode : see ANALYST_PASSCODE in app.py")
    print(f"{'='*72}\n")

    print("Tier distribution - ALL check-ins:")
    print((df["tier"].value_counts(normalize=True) * 100).round(1).to_string(), "\n")

    latest = df.sort_values("checkin").groupby("employee_id").tail(1)
    print("Tier distribution - LATEST check-in per employee (what HR should triage on):")
    print((latest["tier"].value_counts(normalize=True) * 100).round(1).to_string(), "\n")

    print("Employees per department (latest check-in):")
    print(latest["department"].value_counts().to_string(), "\n")

    print("Average risk score by department (latest check-in) - drives the new dept charts:")
    print((latest.groupby("department")["p_high"].mean().sort_values(ascending=False)
           .round(3)).to_string(), "\n")

    print("Job title distribution (one row per employee):")
    print(pd.DataFrame(people, columns=["id", "name", "department", "job_title"])["job_title"]
          .value_counts().to_string(), "\n")

    multi = df.groupby("employee_id").size()
    print(f"Employees with repeat check-ins: {(multi > 1).sum()} of {len(multi)}")
    print(f"Check-ins per employee: min {multi.min()}, max {multi.max()}, mean {multi.mean():.1f}")
    print("\nSanity check - P(High) should FALL as simulated wellbeing rises:")
    print(df.assign(band=pd.cut(df["wellbeing"], [0, .25, .5, .75, 1.0],
                               labels=["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"]))
            .groupby("band", observed=True)["p_high"].agg(["mean", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()