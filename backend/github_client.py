# backend/github_client.py
import os
from github import Github

token = os.getenv("GITHUB_TOKEN")
gh = Github(token)


def get_pr_diff(owner: str, repo: str, pr_number: int) -> str:
    """
    Fetch the unified diff of a pull request.
    """
    repository = gh.get_repo(f"{owner}/{repo}")
    pr = repository.get_pull(pr_number)
    # Collect diffs from files
    diffs = []
    for f in pr.get_files():
        if f.patch:  # unified diff of that file
            diffs.append(f"--- {f.filename}\n+++ {f.filename}\n{f.patch}")
    return "\n\n".join(diffs)
