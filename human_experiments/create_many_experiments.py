import os
import requests
import argparse
import json
from copy import deepcopy

API_TOKEN = os.environ['PROLIFIC_API_TOKEN']
API_URL = "https://api.prolific.com/api/v1"
HEADERS = {
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json",
}

BASE_EXCLUSION = {
    "eligibility_requirements": [
        {
            "type": "prescreener",
            "screener": {
                "type": "exclude_participants_from_other_studies",
                "study_ids": []
            }
        }
    ]
}

def create_and_publish_study(name, access_detail, completion_code, base_config):

    new_config = deepcopy(base_config)
    new_config["completion_codes"][0]['code'] = completion_code
    new_config["name"] = name
    new_config["internal_name"] = name.replace(" ", "_").lower()
    new_config.pop("external_study_url", None)  # Remove if it exists
    new_config["access_details"] = access_detail

    resp = requests.post(f"{API_URL}/studies/", headers=HEADERS, json=new_config)
    resp.raise_for_status()
    study = resp.json()
    stud_id = study["id"]

    # Publish it
    return study


def publish_studies(study):
    # Publish all studies
    try:
        resp = requests.post(f"{API_URL}/studies/{study['id']}/transition/", headers=HEADERS,
                             json={"action": "PUBLISH"})
        resp.raise_for_status()
        print(f"Published study {study['id']} with name {study['name']}")
    except Exception as e:    
        print(f"Failed to publish study {study['id']} with name {study['name']}")


def block_participants_between(study, previous_ids = []):
    # Let’s build block lists so that each study blocks participants from other studies.
    
    filters = study.pop("filters", [])
    for f in filters:
        if f["filter_id"] == "previous_studies_blocklist":
            f['selected_values'] = previous_ids
    
    exclude_data = deepcopy(BASE_EXCLUSION)
    exclude_data["eligibility_requirements"][0]["screener"]["study_ids"] = previous_ids

    resp = requests.patch(f"{API_URL}/studies/{study['id']}/", headers=HEADERS, json={'filters': filters})
    resp.raise_for_status()


def main(pcibex_url, completion_code, num_groups, missing_info=None):

    with open("human_experiments/base_study_config.json", "r") as f:
        base_config = json.load(f)
    
    previous_ids = []
    with open("temp_files/interference_v2_previous_ids.json", "r") as f:
        previous_ids = json.load(f)

    missing_data = {}
    if missing_info:
        with open(missing_info, "r") as f:
            missing_data = json.load(f)

    suffix = "?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}"

    access_details = []

    created = []
    for grp in range(num_groups):
        if missing_data and str(grp) not in missing_data:
            print(f"Skipping group {grp} as it is not in the missing data.")
            continue
        url = f"{pcibex_url.split('?')[0]}" + suffix + f"&withsquare={grp}"
        if not missing_data:
            access_details.append({"external_url": url, "total_allocation": 11})
        else:
            participants = missing_data.get(str(grp), 10) + 2
            access_details.append({"external_url": url, "total_allocation": participants})

    study = create_and_publish_study("Interference v2 full", access_details, completion_code, base_config)
    block_participants_between(study, previous_ids)
    publish_studies(study)
    print("✅ All studies created and cross-blocked.")

    study_ids = [study["id"]]
    previous_ids.extend(study_ids)
    with open("temp_files/interference_v2_previous_ids.json", "w") as f:
        json.dump(previous_ids, f)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Create and publish PCIBex studies on Prolific.")
    parser.add_argument("-u", "--pcibex_url", type=str, required=True, help="The URL of the PCIBex experiment to be used.")
    parser.add_argument("-c", "--completion_code", type=str, required=True, help="Completion code for the study.")
    parser.add_argument("-n", "--num_groups", type=int, default=5, help="Number of groups to create studies for.")
    parser.add_argument("-m", "--missing_info", type=str, default=None,
                        help="Path to a JSON file with missing participant information (optional).")
    args = parser.parse_args()
    main(**vars(args))
