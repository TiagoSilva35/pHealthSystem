import json
import os
from typing import Optional

import requests

BASE_URL = "https://hapi.fhir.org/baseR4"

PATIENT_DB_FILE = os.path.join(os.path.dirname(__file__), "patient_ids.json")

HEADERS = {
    "Content-Type": "application/fhir+json",
    "Accept": "application/fhir+json",
}


def load_patient_ids():

    if os.path.exists(PATIENT_DB_FILE):

        with open(PATIENT_DB_FILE, "r") as f:
            return json.load(f)

    return {}


def save_patient_ids(patient_ids):

    with open(PATIENT_DB_FILE, "w") as f:
        json.dump(patient_ids, f, indent=2)


def post_resource(
    resource_type: str,
    payload: dict,
) -> dict:

    url = f"{BASE_URL}/{resource_type}"

    response = requests.post(
        url,
        headers=HEADERS,
        data=json.dumps(payload),
    )

    print(f"POST {resource_type} -> status {response.status_code}")

    if not response.ok:

        print(response.text)

        if response.status_code == 412:
            print("Resource already exists.")
            return {}

        response.raise_for_status()

    return response.json()


def get_resources(
    resource_type: str,
    params: Optional[dict] = None,
) -> dict:

    url = f"{BASE_URL}/{resource_type}"

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
    )

    print(f"GET {resource_type} -> status {response.status_code}")

    if not response.ok:
        print(response.text)
        response.raise_for_status()

    return response.json()


def get_observation_by_identifier(identifier: str) -> dict:

    return get_resources(
        "Observation",
        params={
            "identifier": (
                f"http://phealth.example.org/observation-id|{identifier}"
            )
        },
    )


def get_patient_by_identifier(identifier: str) -> dict:

    return get_resources(
        "Patient",
        params={
            "identifier": (
                f"http://phealth.example.org/patient-id|{identifier}"
            )
        },
    )


def build_patient_payload(case_data: dict) -> dict:

    return {
        "resourceType": "Patient",
        "identifier": [
            {
                "system": "http://phealth.example.org/patient-id",
                "value": case_data["identifier"],
            }
        ],
        "name": [
            {
                "family": case_data["family_name"],
                "given": [case_data["given_name"]],
            }
        ],
        "gender": case_data["gender"],
        "birthDate": case_data["birth_date"],
        "telecom": [
            {
                "system": "phone",
                "value": case_data["phone"],
                "use": "mobile",
            },
            {
                "system": "email",
                "value": case_data["email"],
                "use": "home",
            },
        ],
        "address": [
            {
                "line": [case_data["address_line"]],
                "city": case_data["city"],
                "country": case_data["country"],
            }
        ],
    }


def build_pvc_per_hour_observation(
    patient_ref: str,
    timestamp: str,
    pvc_per_hour: int,
    obs_identifier: Optional[str] = None,
) -> dict:

    obs = {
        "resourceType": "Observation",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": (
                            "http://terminology.hl7.org/"
                            "CodeSystem/observation-category"
                        ),
                        "code": "laboratory",
                        "display": "Laboratory",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "8049-9",
                    "display": (
                        "Ventricular ectopic beats [#/time] "
                        "in 24 hour Holter monitor"
                    ),
                }
            ],
            "text": "PVC count per hour",
        },
        "subject": {
            "reference": patient_ref
        },
        "effectiveDateTime": timestamp,
        "valueQuantity": {
            "value": pvc_per_hour,
            "unit": "PVC/h",
            "system": "http://unitsofmeasure.org",
            "code": "/h",
        },
    }

    if obs_identifier:

        obs["identifier"] = [
            {
                "system": "http://phealth.example.org/observation-id",
                "value": obs_identifier,
            }
        ]

    return obs


def create_patients():

    patient_ids = load_patient_ids()

    for case_data in PATIENT_CASES:

        identifier = case_data["identifier"]

        cached_patient_id = patient_ids.get(identifier)

        if cached_patient_id:

            print(
                f"Patient already exists locally: "
                f"{identifier} -> Patient/{cached_patient_id}"
            )

            continue

        existing_patient_result = get_patient_by_identifier(identifier)

        if existing_patient_result.get("total", 0) > 0:

            existing_patient = existing_patient_result["entry"][0]["resource"]
            patient_ids[identifier] = existing_patient.get("id")

            print(
                f"Recovered existing Patient/{patient_ids[identifier]} "
                f"for {identifier}"
            )

            continue

        patient_payload = build_patient_payload(case_data)

        patient_result = post_resource(
            "Patient",
            patient_payload,
        )

        patient_id = patient_result.get("id")

        if not patient_id:
            # Fallback for duplicate/create race: resolve by business identifier.
            existing_patient_result = get_patient_by_identifier(identifier)
            if existing_patient_result.get("total", 0) > 0:
                patient_id = existing_patient_result["entry"][0]["resource"].get("id")

        patient_ids[identifier] = patient_id

        print(
            f"Created Patient/{patient_id} "
            f"for {identifier}"
        )

    save_patient_ids(patient_ids)


def send_messages():

    patient_ids = load_patient_ids()

    for case_data in PATIENT_CASES:

        identifier = case_data["identifier"]

        patient_id = patient_ids.get(identifier)

        if not patient_id:

            existing_patient_result = get_patient_by_identifier(identifier)

            if existing_patient_result.get("total", 0) > 0:
                patient_id = existing_patient_result["entry"][0]["resource"].get("id")
                patient_ids[identifier] = patient_id

                print(
                    f"Recovered Patient/{patient_id} "
                    f"for {identifier}"
                )

            else:

                print(
                    f"No stored patient ID for "
                    f"{identifier}"
                )

                continue

        patient_ref = f"Patient/{patient_id}"

        for message in case_data["messages"]:

            obs_id = (
                f"{patient_id}-"
                f"{message['timestamp']}"
            )

            existing_obs = get_observation_by_identifier(
                obs_id
            )

            if existing_obs.get("total", 0) > 0:

                print(
                    f"Observation already exists: "
                    f"{obs_id}"
                )

                continue

            observation_payload = (
                build_pvc_per_hour_observation(
                    patient_ref,
                    message["timestamp"],
                    message["pvc_per_hour"],
                    obs_identifier=obs_id,
                )
            )

            post_resource(
                "Observation",
                observation_payload,
            )

            print(
                f"Sent observation: "
                f"{obs_id}"
            )

    save_patient_ids(patient_ids)


PATIENT_CASES = [
    {
        "label": "Healthy patient",
        "identifier": "PT-HEALTHY-1",
        "family_name": "Silva",
        "given_name": "Miguel",
        "gender": "male",
        "birth_date": "1998-03-12",
        "phone": "+351910000001",
        "email": "miguel.silva@example.org",
        "address_line": "Rua da Saude 1",
        "city": "Lisbon",
        "country": "PT",
        "messages": [
            {
                "timestamp": "2026-05-12T08:00:00Z",
                "pvc_per_hour": 1,
            },
            {
                "timestamp": "2026-05-12T14:00:00Z",
                "pvc_per_hour": 3,
            },
            {
                "timestamp": "2026-05-12T21:00:00Z",
                "pvc_per_hour": 5,
            },
        ],
    },
    {
        "label": "High PVC burden patient",
        "identifier": "PT-PVC-2",
        "family_name": "Costa",
        "given_name": "Ana",
        "gender": "female",
        "birth_date": "1989-11-25",
        "phone": "+351910000002",
        "email": "ana.costa@example.org",
        "address_line": "Avenida Clinica 20",
        "city": "Porto",
        "country": "PT",
        "messages": [
            {
                "timestamp": "2026-05-12T09:00:00Z",
                "pvc_per_hour": 45,
            },
            {
                "timestamp": "2026-05-12T15:00:00Z",
                "pvc_per_hour": 47,
            },
            {
                "timestamp": "2026-05-12T22:00:00Z",
                "pvc_per_hour": 50,
            },
        ],
    },
]


if __name__ == "__main__":

    print("\nFHIR Simulator")
    print("1 - Create Patients")
    print("2 - Send Observations")

    option = input("\nSelect option: ")

    if option == "1":
        create_patients()

    elif option == "2":
        send_messages()

    else:
        print("Invalid option")