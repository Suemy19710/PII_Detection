from collections import defaultdict

def generate_markdown_report(findings):
    md = []
    md.append("# PII Audit Report\n")

    summary = defaultdict(set)

    for f in findings:
        summary[f["type"]].add(f["value"])

    md.append("## Summary\n")
    for k, v in summary.items():
        md.append(f"- **{k}**: {len(v)} unique\n")

    md.append("\n---\n")

    for f in findings:
        md.append(f"### Conversation: {f['title']}")
        md.append(f"- **Type**: {f['type']}")
        md.append(f"- **Value**: `{f['value']}`")
        md.append(f"- **Snippet**: _{f['snippet']}_\n")

    return "\n".join(md)

def quick_summary(entities):
    """
    Minimal summary for direct text input
    """
    summary = {}
    for e in entities:
        summary.setdefault(e["type"], set()).add(e["value"])

    return {
        k: list(v)
        for k, v in summary.items()
    }