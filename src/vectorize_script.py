import os

from dotenv import load_dotenv
from vectorize_book import data_dir, vectorize_book_and_store_to_db, vectorize_chapters

load_dotenv()

raw_subject = os.getenv("CLASS_SUBJECT_NAME") or os.path.join("class_12", "biology")
if os.path.exists(os.path.join(data_dir, raw_subject)):
    CLASS_SUBJECT_NAME = raw_subject
elif os.path.exists(os.path.join(data_dir, "class_12", raw_subject)):
    CLASS_SUBJECT_NAME = os.path.join("class_12", raw_subject)
else:
    CLASS_SUBJECT_NAME = os.path.join("class_12", "biology")

print("Vectorizing Biology:", CLASS_SUBJECT_NAME)

vectorize_book_and_store_to_db(
    CLASS_SUBJECT_NAME,
    vector_db_name="class_12_biology_vector_db",
    subject_label="Biology (Class 12)",
)

vectorize_chapters(
    CLASS_SUBJECT_NAME,
    subject_label="Biology (Class 12)",
)
