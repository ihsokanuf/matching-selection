import configparser
import csv
import hashlib
import logging
import os
from statistics import median


def read_configuration(config_file):
    """Read the configuration from the specified file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    config = configparser.ConfigParser()
    if not os.path.exists(config_file):
        logging.error(f"Configuration file '{config_file}' does not exist.")
        return None

    config.read(config_file, encoding="utf-8")

    return config


def delete_file_if_exists(file_path):
    """Delete the specified file if it exists.

    Args:
        file_path (str): Path of the file to be deleted.
    """
    try:
        os.remove(file_path)
        print(f"{file_path} has been deleted.")
    except FileNotFoundError:
        print(f"{file_path} does not exist.")


def compute_hash(data):
    """Compute SHA-256 hash for the given data.

    Args:
        data: Data to compute hash for.

    Returns:
        str: Computed SHA-256 hash.
    """
    hasher = hashlib.sha256()
    hasher.update(str(data).encode("utf-8"))
    return hasher.hexdigest()


def read_targets_from_csv(file_path):
    """Read and process target data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: Processed data containing ids, target parameters, and target preferences.
    """
    target_params = []
    target_preferences = {}
    id_dict = {}  # Dictionary to store data using id as key

    if not os.path.exists(file_path):
        logging.error(f"CSV file '{file_path}' does not exist.")
        return None

    logging.info(f"Reading file: {file_path}")  # Log the file name

    # Rest of the code...

    # Read the entire CSV at once
    rows = []
    with open(
        file_path, newline="", encoding=config["ENCODING"]["input_encoding"]
    ) as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Get the header row
        rows = []
        for row in reader:
            # Create a dictionary by zipping headers and row values, and add column number as the key
            rows.append(
                {
                    f"{i}_{header}": value
                    for i, (header, value) in enumerate(
                        zip(headers, row), start=1
                    )
                }
            )
        data = {i: row for i, row in enumerate(rows, start=1)}

    # Calculate medians and max values for normalization
    medians = {}
    max_values = {}
    for param in config["PARAMETERS"]:
        column_value = config["PARAMETERS"][param]
        if not column_value:
            continue
        column_index = int(column_value) - 1
        values = [int(row[list(row.keys())[column_index]]) for row in rows]
        medians[param] = median(values)
        max_values[param] = max(values)

    # Process CSV data
    for row in rows:
        id_values = [
            list(row.values())[int(config["IDS"][f"id{i}"]) - 1]
            for i in range(1, len(config["IDS"]) + 1)
            if config["IDS"][f"id{i}"]
        ]
        target_id = "_".join(id_values)
        row["id"] = target_id  # Add ID
        id_dict[target_id] = row  # Save data to dictionary

        target = {"id": target_id}

        for param in config["PARAMETERS"]:
            column_value = config["PARAMETERS"][param]
            if not column_value:
                continue
            column_index = int(column_value) - 1
            if 0 <= column_index < len(row):
                value = int(list(row.values())[column_index])
                normalized_value = (value - medians[param]) / max_values[param]
                target[param] = normalized_value

        target_hash = compute_hash(target)
        target["hash"] = target_hash

        target_params.append(target)
        choices = [
            list(row.values())[int(config["CHOICES"][f"choice{i}"]) - 1]
            for i in range(1, len(config["CHOICES"]) + 1)
            if config["CHOICES"][f"choice{i}"]
        ]
        target_preferences[target_id] = [choice for choice in choices if choice]

        # Assuming calculate_scores_for_recruits function is defined elsewhere
        target_params = calculate_scores_for_recruits(target_params)

    return id_dict, target_params, target_preferences


def evaluate_target(target, section_name="WEIGHTS"):
    """Evaluate a target based on the provided weights.

    Args:
        target (dict): The target data to evaluate.
        section_name (str): The section name in the config where the weights are defined.

    Returns:
        float: The evaluation score for the target.
    """
    score = 0
    for key in config[section_name]:
        weight_value = config[section_name][key]
        if not weight_value:
            continue

        param_key = "param" + key[-1]
        if param_key in target:
            score += target[param_key] * float(weight_value)

    return score


def calculate_scores_for_recruits(targets):
    """Calculate scores for recruits based on their target parameters.

    Args:
        targets (list of dicts): The list of targets to calculate scores for.

    Returns:
        list of dicts: The list of targets with their scores calculated.
    """
    for target in targets:
        for position in config["RECRUIT"]:
            section_name = position + "WEIGHTS"
            score = evaluate_target(target, section_name)
            target[position + "SCORE"] = score

    return targets


def stable_matching(target_preferences, target_params_dict, recruiting_numbers):
    """Perform a stable matching algorithm based on target preferences and recruiting numbers.

    Args:
        target_preferences (dict): The preferences of the targets.
        target_params_dict (dict): The parameters of the targets.
        recruiting_numbers (dict): The number of recruits required for each role.

    Returns:
        dict: The final matching results.
    """
    matches = {role: [] for role in recruiting_numbers}
    unassigned = list(target_preferences.keys())

    while unassigned:
        target = unassigned.pop(0)

        for preference in target_preferences[target]:
            if len(matches[preference]) < recruiting_numbers[preference]:
                matches[preference].append(target)
                break
            else:
                min_score_index = None
                min_score = float("inf")
                min_hash = 0

                for index, matched_target in enumerate(matches[preference]):
                    score = target_params_dict[matched_target][
                        preference + "SCORE"
                    ]
                    target_hash = target_params_dict[matched_target]["hash"]

                    if score < min_score:
                        min_score = score
                        min_score_index = index
                        min_hash = target_hash

                if (
                    target_params_dict[target][preference + "SCORE"] > min_score
                ) or (
                    target_params_dict[target][preference + "SCORE"]
                    == min_score
                    and int(target_params_dict[target]["hash"], 16)
                    > int(min_hash, 16)
                ):
                    unassigned.append(matches[preference][min_score_index])
                    matches[preference][min_score_index] = target
                    break

    return matches


def remove_matched_targets_from_preferences(final_result, target_preferences):
    """Remove matched targets from preferences.

    Args:
        final_result (dict): The final matching results.
        target_preferences (dict): The target preferences.

    Returns:
        dict: Updated target preferences with matched targets removed.
    """
    matched_ids = [id for ids in final_result.values() for id in ids]
    for matched_id in matched_ids:
        target_preferences.pop(matched_id, None)
    return target_preferences


def merge_dicts_with_final_result(
    target_org, target_params, final_result, final_result2
):
    """Merge dictionaries with final matching results.

    Args:
        target_org (dict): Original target data.
        target_params (dict): Target parameters.
        final_result (dict): Final matching results.
        final_result2 (dict): Alternate final matching results.

    Returns:
        dict: Merged data.
    """
    merged = {}
    reversed_final_result = {
        id: role for role, ids in final_result.items() for id in ids
    }
    reversed_final_result2 = {
        id: role for role, ids in final_result2.items() for id in ids
    }

    for target_id, target_data in target_org.items():
        if target_id in target_params:
            merged_data = {**target_data, **target_params[target_id]}
            merged_data["Elected"] = reversed_final_result.get(target_id, "")
            merged_data["Alternate"] = reversed_final_result2.get(target_id, "")
            merged[target_id] = merged_data

    return merged


def merged_data_to_csv(merged, file_path):
    """Write merged data to a CSV file.

    Args:
        merged (dict): The merged data.
        file_path (str): Path to the CSV file to write.
    """
    if not merged:
        print("Warning: Merged data is empty. CSV not written.")
        return

    with open(
        file_path,
        "w",
        newline="",
        encoding=config["ENCODING"]["output_encoding"],
    ) as csvfile:
        fieldnames = list(next(iter(merged.values())).keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged.values():
            writer.writerow(row)


# Logging setup
delete_file_if_exists("matching.log")
logging.basicConfig(
    filename="matching.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Configuration setup
config = read_configuration("config.ini")

# Specifying input and output file names
input_file = config["FILES"]["input_file"]
output_file = config["FILES"]["output_file"]

# Extracting recruiting numbers as dictionary and converting values to integers
recruiting_numbers = {
    key: int(value) for key, value in config["RECRUIT"].items()
}
reserve_numbers = {key: int(value) for key, value in config["RESERVE"].items()}

# Lists representing roles for recruiting and reserve members
recruiting_roles = list(recruiting_numbers.keys())
reserve_roles = list(reserve_numbers.keys())

# Reading target parameters and preferences from 'input_file'
target_org, target_params, target_preferences = read_targets_from_csv(
    input_file
)

# Converting target parameters into a dictionary with ID as the key
target_params_dict = {target["id"]: target for target in target_params}

# Executing the stable matching algorithm and storing the final results
elected_result = stable_matching(
    target_preferences, target_params_dict, recruiting_numbers
)
updated_target_preferences = remove_matched_targets_from_preferences(
    elected_result, target_preferences
)
alternate_result = stable_matching(
    updated_target_preferences, target_params_dict, reserve_numbers
)

# Logging the matching results
for role, target_ids in elected_result.items():
    for target_id in target_ids:
        score = target_params_dict[target_id].get(f"{role}SCORE", 0)
        logging.debug(f"{target_id} is matched with {role}. Score: {score:.2f}")

for role, target_ids in alternate_result.items():
    for target_id in target_ids:
        score = target_params_dict[target_id].get(f"{role}SCORE", 0)
        logging.debug(f"{target_id} is matched with {role}. Score: {score:.2f}")

# Merging the results and writing them to a CSV file
merged_data = merge_dicts_with_final_result(
    target_org, target_params_dict, elected_result, alternate_result
)
merged_data_to_csv(merged_data, output_file)
