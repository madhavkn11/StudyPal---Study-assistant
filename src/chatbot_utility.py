import os

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)


def get_chapter_list(selected_subject):
    if selected_subject == "Operating Systems (B.Tech)":
        return [
            "OS Part 1 - Introduction, OS Structures",
            "OS Part 2 - Processes, Threads",
            "OS Part 3 - CPU Scheduling, Synchronization",
            "OS Part 4 - Deadlocks, Memory Management"
        ]

    subject_map = {
        "Biology": "biology",
        "Physics": "physics",
        "Chemistry": "chemistry"
    }

    subject_name = subject_map.get(selected_subject)

    if subject_name is None:
        return []

    chapters_dir = f"{parent_dir}/data/class_12/{subject_name}"

    chapters_list = [
        file[:-4]
        for file in os.listdir(chapters_dir)
        if file.endswith(".pdf")
    ]

    chapters_list.sort(key=lambda x: int(x.split(".")[0]))

    return chapters_list