import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

N_RESPONSIVE = 600
N_NON_RESPONSIVE = 2400
N_TOTAL = N_RESPONSIVE + N_NON_RESPONSIVE

DOC_TYPES = ["email", "memo", "contract", "financial", "meeting_notes", "hr_document"]
DOC_TYPE_WEIGHTS = [0.45, 0.15, 0.15, 0.10, 0.10, 0.05]

CUSTODIANS = [
    "Jane Hartwell", "Marcus Chen", "Priya Nair", "Tom Dobbins",
    "Sandra Wu", "Derek Okonkwo", "Lisa Fernandez", "Craig Morrow",
    "Yuki Tanaka", "Rachel Bloom", "Alistair Ford", "Neha Sharma",
    "Brian Kowalski", "Diane Tran", "Sergio Reyes",
]

RESPONSIVE_SNIPPETS = [
    "Re: Q3 pricing strategy discussion — see attached amended proposal",
    "Urgent: data retention policy change effective immediately",
    "Attached: amended NDA with Acme Corp — please review before Friday",
    "FWD: Confidential settlement terms — do not distribute",
    "Re: Project Merlin — revised valuation memo attached",
    "Action required: antitrust compliance review of market share data",
    "Re: Exclusive supply agreement — Clause 7 update",
    "PRIVILEGED: Attorney-client communication re: pending litigation",
    "Re: Destruction of records — legal hold notice to follow",
    "Attached: board minutes approving the acquisition of Vantage LLC",
    "Re: Whistleblower complaint — HR investigation findings",
    "Confidential: internal audit report Q2 — do not forward",
    "Re: Side letter agreement with preferred customers",
    "FWD: Rig bidding coordination — Thursday call confirmed",
    "Re: Executive compensation revision — approval pending",
    "Attached: pricing analysis showing competitor rate suppression",
    "Re: Environmental compliance exemption request — urgent",
    "Memo: Retaliatory termination policy discussion",
    "Re: Financial restatement — accounting irregularities identified",
    "Attached: non-disclosure agreement breach — legal action considered",
    "Re: Strategic acquisition targets — Project Eagle shortlist",
    "Executive briefing: regulatory inquiry response strategy",
    "Re: IP ownership dispute with former contractor Chen",
    "FWD: Discriminatory hiring policy — HR flagged for review",
    "Re: Vendor kickback arrangement — invoice discrepancy noted",
    "Restricted: merger agreement draft v4 — not for circulation",
    "Re: Unauthorized data access — IT security incident report",
    "Attached: product liability study — internal use only",
    "Re: Market allocation agreement with Northbridge Partners",
    "Memo: Employee monitoring program expansion — legal sign-off needed",
]

NON_RESPONSIVE_SNIPPETS = [
    "Re: Monday team lunch — Panera or Chipotle?",
    "IT reminder: your password expires in 7 days",
    "Birthday card for Susan in accounting — please sign by Friday",
    "Re: Fantasy football league — week 8 results",
    "FYI: office kitchen cleaned, please label your food",
    "Re: Holiday party planning — venue suggestions welcome",
    "Parking reminder: garage B level 2 reserved for visitors",
    "Re: All-hands meeting — agenda attached",
    "Please welcome David Kim to the Boston office!",
    "IT: Scheduled maintenance window Saturday 2–4 AM",
    "Re: Coffee machine — broken again, sorry team",
    "Reminder: expense reports due by end of month",
    "FYI: building fire drill Wednesday at 10 AM",
    "Re: Ergonomic chair requests — facilities form attached",
    "Annual benefits enrollment closes November 15",
    "Re: Office plants — anyone want to take them home for the holidays?",
    "Congratulations to the sales team on Q3 results!",
    "Re: Dress code reminder for client visit Thursday",
    "IT: New printer driver installed on floor 3",
    "Re: Team outing — kayaking vs. escape room vote",
    "Free training session: Excel pivot tables, Tues 3 PM",
    "Re: Conference room booking — Friday afternoon taken",
    "FYI: Vending machine refilled with healthier options",
    "Reminder: submit your timesheet before noon Friday",
    "Re: Onboarding checklist — new hire starting Monday",
    "IT security tip: do not click suspicious links",
    "Re: Company picnic photos — link to shared album",
    "Quarterly newsletter from HR — summer edition",
    "Re: Update on office expansion to floor 6",
    "Annual performance review schedule now posted on the intranet",
]


def generate():
    doc_ids = [f"DOC-{i+1:04d}" for i in range(N_TOTAL)]

    doc_types = rng.choice(DOC_TYPES, size=N_TOTAL, p=DOC_TYPE_WEIGHTS)
    custodians = rng.choice(CUSTODIANS, size=N_TOTAL)

    start = np.datetime64("2019-01-01")
    end = np.datetime64("2023-12-31")
    days_range = (end - start).astype(int)
    offsets = rng.integers(0, days_range, size=N_TOTAL)
    dates = [str(start + np.timedelta64(int(o), "D")) for o in offsets]

    responsive_snippets_chosen = rng.choice(RESPONSIVE_SNIPPETS, size=N_RESPONSIVE)
    non_responsive_snippets_chosen = rng.choice(NON_RESPONSIVE_SNIPPETS, size=N_NON_RESPONSIVE)
    snippets = list(responsive_snippets_chosen) + list(non_responsive_snippets_chosen)

    ground_truth = [1] * N_RESPONSIVE + [0] * N_NON_RESPONSIVE

    responsive_scores = rng.beta(6, 2, size=N_RESPONSIVE)
    non_responsive_scores = rng.beta(2, 6, size=N_NON_RESPONSIVE)
    confidence_scores = np.concatenate([responsive_scores, non_responsive_scores])
    confidence_scores = np.clip(confidence_scores, 0.0, 1.0).round(4)

    df = pd.DataFrame({
        "doc_id": doc_ids,
        "doc_type": doc_types,
        "custodian": custodians,
        "date": dates,
        "snippet": snippets,
        "ground_truth": ground_truth,
        "confidence_score": confidence_scores,
    })

    # Shuffle so responsive/non-responsive are not contiguous
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = Path(__file__).parent / "documents.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Responsive: {df['ground_truth'].sum()} | Non-responsive: {(df['ground_truth'] == 0).sum()}")
    print(f"Score range: [{df['confidence_score'].min()}, {df['confidence_score'].max()}]")
    print(f"Nulls: {df.isnull().sum().sum()}")


if __name__ == "__main__":
    generate()
