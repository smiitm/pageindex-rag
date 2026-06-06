import groq
from .node import PageNode

client = groq.Groq()


def _summarize(text: str, section_name: str = "") -> str:
    hint = f"This is the section titled: {section_name}.\n" if section_name else ""
    prompt = f"""{hint}Summarize the following in 2-3 sentences. Be specific and factual. Do not add anything not in the text.

{text[:3000]}"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def build_summaries(node: PageNode):
    # post-order: children first
    for child in node.children:
        build_summaries(child)

    if node.is_leaf():
        if node.content.strip():
            node.summary = _summarize(node.content, node.title)
        else:
            node.summary = "(empty section)"
    else:
        # build parent summary from children's summaries
        children_text = "\n\n".join(
            f"[{c.title}]: {c.summary}" for c in node.children
        )
        node.summary = _summarize(children_text, node.title)