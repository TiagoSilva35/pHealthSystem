import json
from datetime import datetime, timezone
from typing import Optional

import requests

BASE_URL = "https://hapi.fhir.org/baseR4"

HEADERS = {
    "Content-Type": "application/fhir+json",
    "Accept": "application/fhir+json",
}


def post_resource(resource_type: str, payload: dict) -> dict:
    url = f"{BASE_URL}/{resource_type}"
    response = requests.post(url, headers=HEADERS, data=json.dumps(payload))
    print(f"POST {resource_type} -> status {response.status_code}")

    if not response.ok:
        print(response.text)
        response.raise_for_status()

    return response.json()


def get_resources(resource_type: str, params: Optional[dict] = None) -> dict:
    url = f"{BASE_URL}/{resource_type}"
    response = requests.get(url, headers=HEADERS, params=params)
    print(f"GET {resource_type} -> status {response.status_code}")

    if not response.ok:
        print(response.text)
        response.raise_for_status()

    return response.json()


def get_patients(identifier: Optional[str] = None) -> dict:
    if identifier is None:
        return get_resources("Patient")
    return get_resources(
        "Patient",
        params={"identifier": f"http://phealth.example.org/patient-id|{identifier}"},
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


def build_pvc_per_hour_observation(patient_ref: str, timestamp: str, pvc_per_hour: int) -> dict:
    return {
        "resourceType": "Observation",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
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
                    "display": "Ventricular ectopic beats [#/time] in 24 hour Holter monitor",
                }
            ],
            "text": "PVC count per hour",
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": timestamp,
        "valueQuantity": {
            "value": pvc_per_hour,
            "unit": "PVC/h",
            "system": "http://unitsofmeasure.org",
            "code": "/h",
        },
    }



def process_patient_case(case_data: dict) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    patient = build_patient_payload(case_data)
    patient_result = post_resource("Patient", patient)
    patient_ref = f"Patient/{patient_result['id']}"

    pvc_per_hour_observation = build_pvc_per_hour_observation(
        patient_ref, timestamp, case_data["pvc_per_hour"]
    )
    
    observation_result = post_resource("Observation", pvc_per_hour_observation)
    fetched_patient_result = get_patients(case_data["identifier"])
    fetched_total = fetched_patient_result.get("total", 0)

    print(
        f"\nFHIR resources for '{case_data['label']}' sent: "
        f"Patient/{patient_result['id']} and Observation/{observation_result['id']}"
    )
    print(
        f"Patient lookup by identifier '{case_data['identifier']}' returned {fetched_total} result(s)."
    )


PATIENT_CASES = [
    {
        "label": "Healthy patient",
        "identifier": "PT-HEALTHY-003",
        "family_name": "Silva",
        "given_name": "Miguel",
        "gender": "male",
        "birth_date": "1998-03-12",
        "phone": "+351910000001",
        "email": "miguel.silva@example.org",
        "address_line": "Rua da Saude 1",
        "city": "Lisbon",
        "country": "PT",
        "pvc_per_hour": 1,
        "has_high_pvc_burden": False,
    },
    {
        "label": "High PVC burden patient",
        "identifier": "PT-PVC-004",
        "family_name": "Costa",
        "given_name": "Ana",
        "gender": "female",
        "birth_date": "1989-11-25",
        "phone": "+351910000002",
        "email": "ana.costa@example.org",
        "address_line": "Avenida Clinica 20",
        "city": "Porto",
        "country": "PT",
        "pvc_per_hour": 45,
        "has_high_pvc_burden": True,
    },
]


for patient_case in PATIENT_CASES:
    process_patient_case(patient_case)