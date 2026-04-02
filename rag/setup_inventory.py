"""
rag/setup_inventory.py

One-time script to create and seed the pharmacy inventory database.

Run once: python rag/setup_inventory.py
Re-run to reset to fresh seed data.

Schema:
  inventory — one row per brand (multiple brands per generic medicine)
  cart      — current session cart (cleared on new session)
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "inventory.db"


SEED_DATA = [
    # (medicine_name, brand_name, strength, form, quantity, unit, price, expiry, schedule, is_promoted, promoted_label)
    # ── Paracetamol ────────────────────────────────────────────────────────
    ("Paracetamol", "Dolo 650",       "650mg",  "tablet",  80,  "strip",  32.0,  "2027-06", "OTC", 1, "Best Seller"),
    ("Paracetamol", "Crocin 500",     "500mg",  "tablet",  150, "strip",  25.0,  "2026-12", "OTC", 0, None),
    ("Paracetamol", "Calpol 250 Syrup","250mg/5ml","bottle",20, "bottle", 55.0,  "2026-09", "OTC", 0, None),

    # ── Ibuprofen ──────────────────────────────────────────────────────────
    ("Ibuprofen",   "Brufen 400",     "400mg",  "tablet",  60,  "strip",  20.0,  "2027-03", "OTC", 0, None),
    ("Ibuprofen",   "Combiflam",      "400mg",  "tablet",  45,  "strip",  28.0,  "2026-11", "OTC", 1, "Recommended"),

    # ── Metformin ──────────────────────────────────────────────────────────
    ("Metformin",   "Glycomet 500",   "500mg",  "tablet",  60,  "strip",   8.0,  "2027-01", "H",   0, None),
    ("Metformin",   "Glucophage 500", "500mg",  "tablet",  0,   "strip",  35.0,  "2026-10", "H",   0, None),  # out of stock

    # ── Antacid / Omeprazole ───────────────────────────────────────────────
    ("Omeprazole",  "Omez 20",        "20mg",   "capsule", 70,  "strip",  22.0,  "2027-04", "OTC", 0, None),
    ("Omeprazole",  "Pan 40",         "40mg",   "tablet",  45,  "strip",  35.0,  "2027-02", "OTC", 1, "Promoted"),
    ("Antacid",     "Digene Syrup",   "liquid", "bottle",  20,  "bottle", 65.0,  "2026-08", "OTC", 0, None),
    ("Antacid",     "Gelusil",        "tablet", "tablet",  30,  "strip",  18.0,  "2026-12", "OTC", 0, None),

    # ── Antihistamine / Allergy ────────────────────────────────────────────
    ("Cetirizine",  "Cetzine 10",     "10mg",   "tablet",  90,  "strip",  15.0,  "2027-05", "OTC", 0, None),
    ("Cetirizine",  "Alerid 10",      "10mg",   "tablet",  50,  "strip",  18.0,  "2027-03", "OTC", 1, "New Stock"),

    # ── ORS ───────────────────────────────────────────────────────────────
    ("ORS",         "Electral Sachet","oral",   "sachet",  100, "sachet",  8.0,  "2027-06", "OTC", 0, None),
    ("ORS",         "Pedialyte",      "oral",   "sachet",  30,  "sachet", 12.0,  "2027-01", "OTC", 1, "Recommended"),

    # ── Antibiotics (Schedule H) ───────────────────────────────────────────
    ("Amoxicillin", "Amoxil 500",     "500mg",  "capsule", 40,  "strip",  45.0,  "2026-11", "H",   0, None),
    ("Azithromycin","Zithromax 500",  "500mg",  "tablet",  30,  "strip",  85.0,  "2026-10", "H",   0, None),
    ("Azithromycin","Azithral 500",   "500mg",  "tablet",  25,  "strip",  72.0,  "2027-02", "H",   1, "Recommended"),

    # ── Vitamins / Supplements ─────────────────────────────────────────────
    ("Vitamin C",   "Limcee 500",     "500mg",  "tablet",  120, "strip",  15.0,  "2027-08", "OTC", 1, "Special Offer"),
    ("Zinc",        "Zincovit",       "tablet", "tablet",  60,  "strip",  22.0,  "2027-04", "OTC", 0, None),
    ("Folic Acid",  "Folvite 5mg",    "5mg",    "tablet",  80,  "strip",  12.0,  "2027-06", "OTC", 0, None),

    # ── Cough / Cold ──────────────────────────────────────────────────────
    ("Cough Syrup", "Benadryl Syrup", "liquid", "bottle",  25,  "bottle", 85.0,  "2026-09", "OTC", 1, "Best Seller"),
    ("Cough Syrup", "Honitus Syrup",  "liquid", "bottle",  15,  "bottle", 75.0,  "2026-11", "OTC", 0, None),

    # ── Amlodipine ────────────────────────────────────────────────────────
    ("Amlodipine",  "Amlovas 5",      "5mg",    "tablet",  55,  "strip",  12.0,  "2027-03", "H",   0, None),
    ("Amlodipine",  "Amlip 5",        "5mg",    "tablet",  40,  "strip",  10.0,  "2027-05", "H",   1, "Recommended"),
]


def setup():
    import csv

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS inventory;
        DROP TABLE IF EXISTS drug_info;
        DROP TABLE IF EXISTS cart;

        CREATE TABLE inventory (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_name  TEXT NOT NULL,
            brand_name     TEXT NOT NULL UNIQUE,
            strength       TEXT,
            form           TEXT,
            quantity       INTEGER DEFAULT 0,
            unit           TEXT DEFAULT 'strip',
            price          REAL NOT NULL,
            expiry         TEXT,
            schedule       TEXT DEFAULT 'OTC',
            is_promoted    INTEGER DEFAULT 0,
            promoted_label TEXT
        );

        CREATE TABLE drug_info (
            generic_name     TEXT PRIMARY KEY,
            category         TEXT,
            uses             TEXT,
            dosage_adult     TEXT,
            dosage_child     TEXT,
            side_effects     TEXT,
            contraindications TEXT,
            interactions     TEXT
        );

        CREATE TABLE cart (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            brand_name  TEXT NOT NULL,
            medicine    TEXT NOT NULL,
            quantity    INTEGER NOT NULL,
            unit_price  REAL NOT NULL,
            form        TEXT
        );
    """)

    cur.executemany("""
        INSERT INTO inventory
          (medicine_name, brand_name, strength, form, quantity, unit, price,
           expiry, schedule, is_promoted, promoted_label)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, SEED_DATA)

    # Load drug info from medicines.csv into drug_info table
    csv_path = Path(__file__).parent / "data" / "medicines.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            drug_rows = [
                (
                    row["name"], row["category"], row["uses"],
                    row["dosage_adult"], row["dosage_child"],
                    row["side_effects"], row["contraindications"], row["interactions"],
                )
                for row in reader
            ]
        cur.executemany("""
            INSERT OR IGNORE INTO drug_info
              (generic_name, category, uses, dosage_adult, dosage_child,
               side_effects, contraindications, interactions)
            VALUES (?,?,?,?,?,?,?,?)
        """, drug_rows)
        print(f"Loaded {len(drug_rows)} drug profiles from medicines.csv")
    else:
        print("WARNING: medicines.csv not found — drug_info table will be empty")

    # Extra drug profiles for medicines in inventory but not in medicines.csv
    EXTRA_DRUG_INFO = [
        # (generic_name, category, uses, dosage_adult, dosage_child, side_effects, contraindications, interactions)
        ("Antacid", "Antacid",
         "Stomach acidity; heartburn; indigestion; stomach ache; acid reflux; stomach pain",
         "1-2 tablets or 10ml after meals and at bedtime", "Half adult dose",
         "Constipation (calcium-based); diarrhea (magnesium-based)", "Kidney disease", ""),
        ("Cough Syrup", "Antitussive / Expectorant",
         "Dry cough; wet cough; chest congestion; cold; sore throat; respiratory infection",
         "10ml every 4-6 hours, max 4 doses/day", "5ml every 6-8 hours",
         "Drowsiness; nausea", "Avoid in children under 2", ""),
        ("Vitamin C", "Vitamin / Supplement",
         "Vitamin C deficiency; immune support; cold prevention; wound healing; scurvy",
         "500-1000mg daily", "250mg daily",
         "Stomach upset at high doses; kidney stones (rare)", "Kidney stones history", ""),
        ("Zinc", "Mineral / Supplement",
         "Zinc deficiency; immunity; diarrhea in children; wound healing; growth",
         "10-25mg daily", "10mg daily for 10-14 days (diarrhea)",
         "Nausea; metallic taste at high doses", "Avoid excessive doses", ""),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO drug_info
          (generic_name, category, uses, dosage_adult, dosage_child,
           side_effects, contraindications, interactions)
        VALUES (?,?,?,?,?,?,?,?)
    """, EXTRA_DRUG_INFO)
    print(f"Added {len(EXTRA_DRUG_INFO)} extra drug profiles.")

    conn.commit()
    conn.close()
    print(f"Inventory database created at {DB_PATH}")
    print(f"Seeded {len(SEED_DATA)} products.")


if __name__ == "__main__":
    setup()
