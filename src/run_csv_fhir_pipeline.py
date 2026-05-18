#!/usr/bin/env python3
"""Optional ECG CSV -> feature extraction -> FHIR bridge."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.FHIR import (
    build_patient_payload,
    build_pvc_per_hour_observation,
    get_observation_by_identifier,
    get_patient_by_identifier,
    load_patient_ids,
    post_resource,
    save_patient_ids,
)
from src.run_mitdb import evaluate_ecg_csv


def prompt_text(label, default=None, required=False, allowed=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{label}{suffix}: ").strip()
        if not value and default is not None:
            value = str(default)
        if required and not value:
            print("This field is required.")
            continue
        if allowed and value and value not in allowed:
            print(f"Allowed values: {', '.join(allowed)}")
            continue
        return value


def prompt_yes_no(label, default=False):
    default_token = "Y/n" if default else "y/N"
    while True:
        value = input(f"{label} [{default_token}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer with y or n.")


def parse_age_to_birthdate(age_text):
    try:
        age = int(age_text)
    except ValueError:
        raise ValueError("Age must be an integer.")
    if age < 0 or age > 130:
        raise ValueError("Age must be between 0 and 130.")
    year = datetime.now(timezone.utc).year - age
    return f"{year:04d}-01-01"


def build_patient_case_from_form(identifier):
    print("\nPatient form")
    given_name = prompt_text("Given name", required=True)
    family_name = prompt_text("Family name", required=True)
    gender = prompt_text("Gender (male/female/other/unknown)", default="unknown", allowed={"male", "female", "other", "unknown"})

    birth_date = prompt_text("Birth date (YYYY-MM-DD, optional)")
    if not birth_date:
        while True:
            age_text = prompt_text("Age (optional)")
            if not age_text:
                birth_date = "1970-01-01"
                break
            try:
                birth_date = parse_age_to_birthdate(age_text)
                break
            except ValueError as exc:
                print(exc)

    phone = prompt_text("Phone", default="+351910000000")
    email = prompt_text("Email", default=f"{identifier.lower()}@example.org")
    address_line = prompt_text("Address line", default="Unknown")
    city = prompt_text("City", default="Unknown")
    country = prompt_text("Country (ISO 2-letter)", default="PT")

    return {
        "identifier": identifier,
        "given_name": given_name,
        "family_name": family_name,
        "gender": gender,
        "birth_date": birth_date,
        "phone": phone,
        "email": email,
        "address_line": address_line,
        "city": city,
        "country": country,
    }


def ensure_patient_resource(case_data):
    patient_ids = load_patient_ids()
    identifier = case_data["identifier"]

    cached_patient_id = patient_ids.get(identifier)
    if cached_patient_id:
        print(f"Patient already exists locally: {identifier} -> Patient/{cached_patient_id}")
        return cached_patient_id

    existing_patient_result = get_patient_by_identifier(identifier)
    if existing_patient_result.get("total", 0) > 0:
        patient_id = existing_patient_result["entry"][0]["resource"].get("id")
        patient_ids[identifier] = patient_id
        save_patient_ids(patient_ids)
        print(f"Recovered existing Patient/{patient_id} for {identifier}")
        return patient_id

    patient_payload = build_patient_payload(case_data)
    patient_result = post_resource("Patient", patient_payload)
    patient_id = patient_result.get("id")

    if not patient_id:
        existing_patient_result = get_patient_by_identifier(identifier)
        if existing_patient_result.get("total", 0) > 0:
            patient_id = existing_patient_result["entry"][0]["resource"].get("id")

    if not patient_id:
        raise RuntimeError("Could not create or recover Patient resource ID.")

    patient_ids[identifier] = patient_id
    save_patient_ids(patient_ids)
    print(f"Created Patient/{patient_id} for {identifier}")
    return patient_id


def resolve_patient_id_without_creation(identifier):
    patient_ids = load_patient_ids()
    patient_id = patient_ids.get(identifier)
    if patient_id:
        return patient_id

    existing_patient_result = get_patient_by_identifier(identifier)
    if existing_patient_result.get("total", 0) > 0:
        patient_id = existing_patient_result["entry"][0]["resource"].get("id")
        patient_ids[identifier] = patient_id
        save_patient_ids(patient_ids)
        print(f"Recovered Patient/{patient_id} for {identifier}")
        return patient_id

    raise RuntimeError(
        f"No patient found for identifier '{identifier}'. "
        "Run again with --create-patient to create it first."
    )


def format_obs_identifier(patient_id, csv_path, timestamp):
    ts = timestamp.replace(":", "").replace("+", "").replace("-", "").replace(".", "")
    return f"{patient_id}-{Path(csv_path).stem}-{ts}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optional bridge: run ECG CSV detection and optionally send FHIR Patient/Observation."
    )
    parser.add_argument("--ecg-csv", required=True, help="Path to ECG CSV file with columns time_s,ecg")
    parser.add_argument("--output", default="mitdb_results", help="Output directory for extracted features")
    parser.add_argument("--skip-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--min-peak-distance", type=float, default=0.06)
    parser.add_argument("--refractory", type=float, default=0.12)
    parser.add_argument("--prematurity-threshold", type=float, default=0.85)
    parser.add_argument("--qrs-width-threshold-ms", type=float, default=120.0)
    parser.add_argument(
        "--detection-rule",
        choices=["and", "or", "weighted", "mlp", "2of4"],
        default="and",
        help="PVC detection rule to pass to run_mitdb extractor",
    )
    parser.add_argument("--create-patient", action="store_true", help="Prompt for patient form and create/recover Patient in FHIR")
    parser.add_argument("--create-observation", action="store_true", help="Optionally create Observation in FHIR using computed PVC/hour")
    parser.add_argument("--patient-identifier", default=None, help="Business identifier to reuse when creating observation")
    parser.add_argument("--observation-time", default=None, help="effectiveDateTime for Observation (default: current UTC)")
    return parser.parse_args()


def main():
    args = parse_args()

    summary, features, times, _ = evaluate_ecg_csv(
        csv_path=args.ecg_csv,
        output_dir=args.output,
        skip_plots=args.skip_plots,
        show_plots=args.show,
        min_peak_distance_s=args.min_peak_distance,
        refractory_s=args.refractory,
        prematurity_threshold=args.prematurity_threshold,
        qrs_width_threshold_ms=args.qrs_width_threshold_ms,
        detection_rule=args.detection_rule,
    )

    pvc_count = int(sum(int(row["is_pvc_candidate"]) for row in features))
    duration_s = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    pvc_per_hour = (3600.0 * pvc_count / duration_s) if duration_s > 0 else 0.0

    print("\nFHIR optional step")
    print(f"- PVC candidates in file: {pvc_count}")
    print(f"- File duration (s): {duration_s:.2f}")
    print(f"- Computed PVC/hour: {pvc_per_hour:.2f}")

    should_create_patient = args.create_patient
    should_create_observation = args.create_observation
    if not (should_create_patient or should_create_observation):
        should_create_patient = prompt_yes_no("Create/recover Patient in FHIR?", default=False)
        should_create_observation = prompt_yes_no("Create Observation in FHIR?", default=False)

    if not (should_create_patient or should_create_observation):
        print("FHIR creation skipped.")
        return

    identifier = args.patient_identifier or prompt_text("Patient identifier", required=True)
    patient_id = None

    if should_create_patient:
        case_data = build_patient_case_from_form(identifier)
        patient_id = ensure_patient_resource(case_data)

    if should_create_observation:
        if not patient_id:
            patient_id = resolve_patient_id_without_creation(identifier)
        patient_ref = f"Patient/{patient_id}"
        timestamp = args.observation_time or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        obs_id = format_obs_identifier(patient_id, args.ecg_csv, timestamp)

        existing_obs = get_observation_by_identifier(obs_id)
        if existing_obs.get("total", 0) > 0:
            print(f"Observation already exists: {obs_id}")
            return

        observation_payload = build_pvc_per_hour_observation(
            patient_ref,
            timestamp,
            round(pvc_per_hour, 2),
            obs_identifier=obs_id,
        )
        post_resource("Observation", observation_payload)
        print(f"Created Observation {obs_id} (PVC/h={pvc_per_hour:.2f})")

    print(f"Done. Summary file: {Path(args.output) / (Path(args.ecg_csv).stem + '_csv_eval.csv')}")
    print(f"Run metrics reused from run_mitdb summary: {summary['record']}")


if __name__ == "__main__":
    main()
