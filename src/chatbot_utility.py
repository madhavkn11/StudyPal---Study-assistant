import os
import re
from typing import List, Optional

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

OS_CHAPTER_LABELS = [
    "All Chapters",
    "OS Part 1 - Introduction, OS Structures",
    "OS Part 2 - OS Structures, Processes",
    "OS Part 3 - Processes, Threads, Scheduling",
    "OS Part 4 - Scheduling, Synchronization, Deadlocks",
]


def _natural_sort_key(text: str):
    """Sort chapter names naturally (e.g., Ch 6 before Ch 12)."""
    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", text)
    ]


BIOLOGY_CHAPTER_TITLES = {
    "Ch 6": "Ch 6 - Evolution",
    "Ch 12": "Ch 12 - Ecosystem",
}


def get_chapter_list(selected_subject: str) -> List[str]:
    if selected_subject == "Operating Systems (B.Tech)":
        return list(OS_CHAPTER_LABELS)

    subject_map = {
        "Biology": "biology",
        "Biology (Class 12)": "biology",
        "Physics": "physics",
        "Physics (Class 12)": "physics",
        "Chemistry": "chemistry",
        "Chemistry (Class 12)": "chemistry",
    }

    subject_name = subject_map.get(selected_subject)
    if subject_name is None:
        return []

    chapters_dir = os.path.join(parent_dir, "data", "class_12", subject_name)
    if not os.path.exists(chapters_dir):
        return []

    chapters_list = [
        file[:-4]
        for file in os.listdir(chapters_dir)
        if file.lower().endswith(".pdf")
    ]
    chapters_list.sort(key=_natural_sort_key)
    if subject_name == "biology":
        chapters_list = [BIOLOGY_CHAPTER_TITLES.get(ch, ch) for ch in chapters_list]
    return chapters_list


def map_chapter_to_filter(selected_chapter: Optional[str], selected_subject: str) -> Optional[str]:
    """Map UI chapter display name to exact ChromaDB metadata `chapter` key."""
    if not selected_chapter or selected_chapter == "All Chapters":
        return None
    # "OS Part 1 - Introduction, OS Structures" -> "OS Part 1"
    # "Ch 6 - Evolution" -> "Ch 6"
    return selected_chapter.split(" - ")[0].strip()
