import requests
import json
from datetime import datetime, timezone

BASE_URL = "https://hapi.fhir.org/baseR4"

headers = {
    "Content-Type": "application/fhir+json",
    "Accept": "application/fhir+json"
}


def post_resource(resource_type: str, payload: dict) -> dict:
    url = f"{BASE_URL}/{resource_type}"
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(f"POST {resource_type} -> status {response.status_code}")

    if not response.ok:
        print(response.text)
        response.raise_for_status()

    return response.json()


def get_resource(url: str) -> dict:
    response = requests.get(url, headers=headers)
    print(f"GET {url} -> status {response.status_code}")

    if not response.ok:
        print(response.text)
        response.raise_for_status()

    return response.json()


# ------------------------------------------------------------
# 1) Create patient once
# ------------------------------------------------------------
patient_payload = {
    "resourceType": "Patient",
    "name": [
        {
            "family": "Silva",
            "given": ["Tiago"]
        }
    ],
    "gender": "male",
    "birthDate": "2000-01-01"
}

patient_result = post_resource("Patient", patient_payload)
patient_id = patient_result["id"]
print("Patient ID:", patient_id)

# Optional GET to confirm
patient_get = get_resource(f"{BASE_URL}/Patient/{patient_id}")
print(json.dumps(patient_get, indent=2))


# ------------------------------------------------------------
# 2) Values produced by your algorithm for one monitoring instant
# ------------------------------------------------------------
timestamp = datetime.now(timezone.utc).isoformat()

hr_bpm = 120
pvc_count_in_window = 8
total_beats_in_window = 100
pvc_burden_percent = round((pvc_count_in_window / total_beats_in_window) * 100, 2)

# Your classifier output
diagnosis_text = "Premature ventricular contractions"
diagnosis_status = "provisional" 


# ------------------------------------------------------------
# 3) Heart rate Observation
# ------------------------------------------------------------
hr_observation = {
    "resourceType": "Observation",
    "status": "final",
    "category": [
        {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "vital-signs",
                    "display": "Vital Signs"
                }
            ]
        }
    ],
    "code": {
        "coding": [
            {
                "system": "http://loinc.org",
                "code": "8867-4",
                "display": "Heart rate"
            }
        ]
    },
    "subject": {
        "reference": f"Patient/{patient_id}"
    },
    "effectiveDateTime": timestamp,
    "valueQuantity": {
        "value": hr_bpm,
        "unit": "beats/minute",
        "system": "http://unitsofmeasure.org",
        "code": "/min"
    }
}

hr_result = post_resource("Observation", hr_observation)
print("HR Observation ID:", hr_result["id"])


# ------------------------------------------------------------
# 4) PVC count Observation
# ------------------------------------------------------------
pvc_count_observation = {
    "resourceType": "Observation",
    "status": "final",
    "category": [
        {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "laboratory",
                    "display": "Laboratory"
                }
            ]
        }
    ],
    "code": {
        "text": "PVC count in monitoring window"
    },
    "subject": {
        "reference": f"Patient/{patient_id}"
    },
    "effectiveDateTime": timestamp,
    "valueQuantity": {
        "value": pvc_count_in_window,
        "unit": "PVCs"
    }
}

pvc_count_result = post_resource("Observation", pvc_count_observation)
print("PVC Count Observation ID:", pvc_count_result["id"])


# ------------------------------------------------------------
# 5) PVC burden Observation
# ------------------------------------------------------------
pvc_burden_observation = {
    "resourceType": "Observation",
    "status": "final",
    "category": [
        {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "survey",
                    "display": "Survey"
                }
            ]
        }
    ],
    "code": {
        "text": "PVC burden"
    },
    "subject": {
        "reference": f"Patient/{patient_id}"
    },
    "effectiveDateTime": timestamp,
    "valueQuantity": {
        "value": pvc_burden_percent,
        "unit": "%"
    }
}

pvc_burden_result = post_resource("Observation", pvc_burden_observation)
print("PVC Burden Observation ID:", pvc_burden_result["id"])


# ------------------------------------------------------------
# 6) Condition for the algorithmic diagnosis / screening result
# ------------------------------------------------------------
condition_payload = {
    "resourceType": "Condition",
    "clinicalStatus": {
        "coding": [
            {
                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                "code": "active"
            }
        ]
    },
    "verificationStatus": {
        "coding": [
            {
                "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                "code": diagnosis_status
            }
        ]
    },
    "category": [
        {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                    "code": "problem-list-item"
                }
            ]
        }
    ],
    "code": {
        "text": diagnosis_text
    },
    "subject": {
        "reference": f"Patient/{patient_id}"
    },
    "recordedDate": timestamp
}

condition_result = post_resource("Condition", condition_payload)
print("Condition ID:", condition_result["id"])


# ------------------------------------------------------------
# 7) Useful GETs for validation
# ------------------------------------------------------------
obs_bundle = get_resource(f"{BASE_URL}/Observation?subject=Patient/{patient_id}")
print(json.dumps(obs_bundle, indent=2))

cond_bundle = get_resource(f"{BASE_URL}/Condition?subject=Patient/{patient_id}")
print(json.dumps(cond_bundle, indent=2))