import argparse
from pathlib import Path

from src.service import SPBEService


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SPBE text-response prototype.")
    parser.add_argument("--target", default="said", help="Expected target word")
    parser.add_argument("--response", default="side", help="Learner response")
    parser.add_argument("--learner", default="demo_learner", help="Non-identifying learner ID")
    parser.add_argument(
        "--profile-file",
        type=Path,
        default=Path("outputs") / "learner_profiles.json",
        help="JSON file used for the prototype learner profile",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    result = SPBEService(args.profile_file).analyse_attempt(
        args.target, args.response, args.learner
    )
    analysis = result["analysis"]
    intervention = result["intervention"]
    profile = result["profile"]

    print("SEMANTIC PHONETIC BRIDGE ENGINE")
    print(f"Target: {analysis['target']}")
    print(f"Learner response: {analysis['response']}")
    print(f"Correct response: {analysis['is_correct']}")
    print(f"Modality scores: {analysis['modality_scores']}")
    print(f"Bridge scores: {analysis['bridge_scores']}")
    print(f"Detected weak bridge: {analysis['weak_mapping']}")
    print(f"Suggested intervention: {intervention['title']}")
    for activity in intervention["activities"]:
        print(f"- {activity}")
    print(f"Profile attempts for {args.learner}: {profile['attempts']}")


if __name__ == "__main__":
    main()
