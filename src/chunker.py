
import json
import re
from pathlib import Path

# ==================================================
# CONFIGURATION
# ==================================================

INPUT_FILE = Path("data/portfolio_text.txt")
OUTPUT_FILE = Path("data/chunks.json")

CHUNK_SIZE = 120
CHUNK_OVERLAP = 25

PROJECTS = [
    "ML Model Monitoring & Drift Detection",
    "Agentic Compliance Auditor",
    "MindTrace AI",
    "MatRisk AI",
    "IBM Customer Churn MLOps",
    "MarketMinds",
    "GroMart",
]

IGNORE_LINES = {
    "View",
    "View case",
    "case",
    "Scroll",
    "Case index",
    "Back to top ↑",
    "✕",
    "Project Work",
    "Experience",
    "Research",
    "Gallery",
    "Capabilities",
    "About",
    "Contact",
}


# ==================================================
# READ SOURCE
# ==================================================

def read_lines():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Source file not found: {INPUT_FILE}"
        )

    return [
        line.strip()
        for line in INPUT_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


# ==================================================
# CLEAN TEXT
# ==================================================

def clean_lines(lines):

    cleaned = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line in IGNORE_LINES:
            continue

        # Remove carousel numbering
        if re.match(r"^\d+\s*/\s*\d+", line):
            continue

        # Remove decorative numbering
        if re.match(r"^\d+\s*—\s*\d+$", line):
            continue

        cleaned.append(line)

    return cleaned


# ==================================================
# SECTION BOUNDARIES
# ==================================================

def find_line(lines, value, start=0):

    for index in range(start, len(lines)):

        if lines[index] == value:
            return index

    return -1


def find_last_line(lines, value):

    for index in range(len(lines) - 1, -1, -1):

        if lines[index] == value:
            return index

    return -1


def extract_between(lines, start, end):

    start_index = find_last_line(lines, start)
    end_index = find_last_line(lines, end)

    if start_index == -1 or end_index == -1:
        return []

    if end_index <= start_index:
        return []

    return lines[start_index + 1:end_index]


# ==================================================
# TEXT CHUNKING
# ==================================================

def split_into_chunks(text):

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + CHUNK_SIZE,
            len(words)
        )

        chunks.append(
            " ".join(words[start:end])
        )

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ==================================================
# CREATE RECORD
# ==================================================

def make_record(
    text,
    section,
    title,
    index
):

    safe_title = (
        title.lower()
        .replace(" ", "_")
        .replace("&", "and")
    )

    return {
        "chunk_id": (
            f"{section}_{safe_title}_{index:03d}"
        ),
        "section": section,
        "title": title,
        "source": "atharvbhosale.live",
        "text": text
    }


# ==================================================
# PROJECT EXTRACTION
# ==================================================

def extract_projects(lines):

    records = []

    project_start = find_line(
        lines,
        "Project Work"
    )

    project_end = find_line(
        lines,
        "Experience",
        project_start + 1
    )

    if project_start == -1 or project_end == -1:
        raise ValueError(
            "Could not identify Project Work boundaries"
        )

    project_positions = []

    for index in range(
        project_start + 1,
        project_end
    ):

        if lines[index] in PROJECTS:

            project_positions.append(
                (index, lines[index])
            )

    if not project_positions:
        raise ValueError(
            "No projects found"
        )

    # Extract each project using its card boundary.
    for i, (start, project) in enumerate(
        project_positions
    ):

        # Find the next project card's category marker.
        end = project_end

        for next_index in range(
            start + 1,
            project_end
        ):

            if lines[next_index].startswith("— "):

                end = next_index
                break

        project_lines = clean_lines(
            lines[start:end]
        )

        project_text = " ".join(
            project_lines
        )

        # Attach expanded ML Monitoring details.
        if project == "ML Model Monitoring & Drift Detection":

            occurrences = [
                index
                for index, line in enumerate(lines)
                if line == project
            ]

            detail_positions = [
                index
                for index in occurrences
                if index > project_end
            ]

            if detail_positions:

                detail_start = detail_positions[0]

                # Expanded details extend to the end
                # of the extracted portfolio document.
                detail_lines = clean_lines(
                    lines[detail_start:]
                )

                detail_text = " ".join(
                    detail_lines
                )

                project_text += " " + detail_text

        # Generate independent project chunks.
        chunks = split_into_chunks(
            project_text
        )

        for j, chunk in enumerate(chunks, 1):

            contextual_text = (
                f"Project: {project}. {chunk}"
            )

            records.append(
                make_record(
                    contextual_text,
                    "projects",
                    project,
                    j
                )
            )

    return records


# ==================================================
# GENERAL SECTION EXTRACTION
# ==================================================

def extract_section(
    lines,
    start,
    end,
    section,
    title
):

    section_lines = extract_between(
        lines,
        start,
        end
    )

    section_lines = clean_lines(
        section_lines
    )

    if not section_lines:
        print(
            f"WARNING: No content found for {title}"
        )
        return []

    section_text = " ".join(
        section_lines
    )

    chunks = split_into_chunks(
        section_text
    )

    records = []

    for index, chunk in enumerate(chunks, 1):

        records.append(
            make_record(
                f"{title}: {chunk}",
                section,
                title,
                index
            )
        )

    return records


# ==================================================
# MAIN PIPELINE
# ==================================================

def build_chunks():

    lines = read_lines()

    records = []

    # 1. Projects
    records.extend(
        extract_projects(lines)
    )

    # 2. Experience
    records.extend(
        extract_section(
            lines,
            "Experience",
            "Research",
            "experience",
            "Professional Experience"
        )
    )

    # 3. Research
    records.extend(
        extract_section(
            lines,
            "Research",
            "Gallery",
            "research",
            "Research Publications"
        )
    )

    # 4. Achievements
    records.extend(
        extract_section(
            lines,
            "Gallery",
            "Capabilities",
            "achievements",
            "Achievements"
        )
    )

    # 5. Skills
    records.extend(
        extract_section(
            lines,
            "Capabilities",
            "About",
            "skills",
            "Technical Skills"
        )
    )

    # 6. About
    records.extend(
        extract_section(
            lines,
            "About",
            "Contact",
            "about",
            "About Atharv"
        )
    )

    # Contact details intentionally excluded.

    # Validate unique IDs
    ids = [
        record["chunk_id"]
        for record in records
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate chunk IDs detected!"
        )

    # Save chunks
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            records,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # ==================================================
    # REPORT
    # ==================================================

    print("\n========== CHUNKING COMPLETED ==========")

    print("Total chunks:", len(records))

    print("\n========== SECTION DISTRIBUTION ==========")

    distribution = {}

    for record in records:

        section = record["section"]

        distribution[section] = (
            distribution.get(section, 0) + 1
        )

    for section, count in distribution.items():

        print(f"{section}: {count}")

    print("\n========== PROJECT VERIFICATION ==========")

    for record in records:

        if record["section"] == "projects":

            print(
                f"\nProject: {record['title']}"
            )

            print(
                f"Chunk ID: {record['chunk_id']}"
            )

            print(
                f"Words: {len(record['text'].split())}"
            )

            print(
                f"Content: {record['text'][:350]}"
            )

    print("\nSaved to:", OUTPUT_FILE)


# ==================================================
# EXECUTION
# ==================================================

if __name__ == "__main__":

    build_chunks()

