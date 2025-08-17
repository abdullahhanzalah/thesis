import os
import random


def create_dataset_paths(dataset_path, total_slices=50, patient_nums=None):
    es_count, ed_count = 0, 0
    train_paths = []

    new_random_numbers = [random.randint(1, 100) for _ in range(10)]
    if not patient_nums:
        patient_nums = [f"{num:03}" for num in new_random_numbers]

    print(f"NUMS: {patient_nums}")

    for patient_num in patient_nums:
        if es_count + ed_count >= total_slices:
            print("Reached 50 slices, stopping.")
            break

        for i in range(0, 15):
            slice_es_path = os.path.join(
                dataset_path, f"patient{patient_num}_SA_ES_slice_{i}.h5"
            )
            if not os.path.exists(slice_es_path):
                continue
            es_count += 1
            train_paths.append(slice_es_path)

        for i in range(0, 15):
            slice_ed_path = os.path.join(
                dataset_path, f"patient{patient_num}_SA_ED_slice_{i}.h5"
            )
            if not os.path.exists(slice_ed_path):
                continue
            ed_count += 1
            train_paths.append(slice_ed_path)

    print(f"Counts - ES: {es_count}, ED: {ed_count}")
    return train_paths


def create_validation_paths(dataset_path, total_slices=50, patient_nums=None):
    files = os.listdir(dataset_path)
    if not patient_nums:
        patient_nums = [file.split("_")[0][-3:]for file in files]
        patient_nums = list(set(patient_nums))  # Unique patient numbers
        random.shuffle(patient_nums)  # Shuffle the patient numbers

    validation_paths = []
    es_count, ed_count = 0, 0
    for patient_num in patient_nums:
        if es_count + ed_count >= total_slices:
            print("Reached 50 slices, stopping.")
            break

        for i in range(0, 15):
            slice_es_path = os.path.join(
                dataset_path, f"patient{patient_num}_SA_ES_slice_{i}.h5"
            )
            if not os.path.exists(slice_es_path):
                continue
            es_count += 1
            validation_paths.append(slice_es_path)

        for i in range(0, 15):
            slice_ed_path = os.path.join(
                dataset_path, f"patient{patient_num}_SA_ED_slice_{i}.h5"
            )
            if not os.path.exists(slice_ed_path):
                continue
            ed_count += 1
            validation_paths.append(slice_ed_path)

    print(f"Counts - ES: {es_count}, ED: {ed_count}")
    return validation_paths
