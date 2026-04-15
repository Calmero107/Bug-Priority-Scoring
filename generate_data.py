from pathlib import Path

from src.bug_priority.data_generation import GenerationConfig, save_dataset


if __name__ == "__main__":
    output = save_dataset(Path("data") / "bugs.csv", GenerationConfig(rows=1500))
    print(f"Dataset generated at: {output}")
