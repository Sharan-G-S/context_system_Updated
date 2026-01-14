# cli_chat.py
import os

# Silence tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from chat_service import add_super_chat_message
from context_builder import build_context
from llm import call_llm

from file_ingestion import ingest_markdown
from markdown_utils import count_tokens
from redis_stm_index import create_index

USER_ID = "11111111-1111-1111-1111-111111111111"
TOKEN_THRESHOLD = 10


def run_cli():
    # Initialize Redis STM index if not exists
    create_index()
    print("🧠 Context-Aware Chat (CLI)")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("👋 Goodbye")
            break

        # 1️⃣ Store user message (always)
        add_super_chat_message(USER_ID, "user", user_input)

        # 2️⃣ Long-term file ingestion (write-time)
        token_count = count_tokens(user_input)
        if token_count > TOKEN_THRESHOLD:
            print(f"📥 Long input detected ({token_count} tokens)")
            file_id = ingest_markdown(
                user_id=USER_ID,
                filename="cli_input.md",
                markdown=user_input
            )
            if file_id:
                print(f"📄 Stored as long-term file memory ({file_id})")

        # 3️⃣ Build context (STM → File + Episodic → Dedup → Rerank → Compress)
        context = build_context(USER_ID, user_input)

        # 4️⃣ Print FINAL context that will be sent to LLM
        if context:
            print(f"\n────────── 🔎 FINAL CONTEXT SENT TO LLM ({len(context)}) ──────────")
            for i, ctx in enumerate(context, 1):
                print(f"[CONTEXT {i}]")
                print(ctx["content"])
                print()
            print("──────────────────────────────────────────────────────────────\n")
        else:
            print("🔍 No relevant context retrieved.\n")

        # 5️⃣ LLM call
        messages = [
            {"role": ctx["role"], "content": ctx["content"]}
            for ctx in context
        ]
        messages.append({
            "role": "user",
            "content": user_input
        })

        response = call_llm(messages)

        # 6️⃣ Store assistant reply
        add_super_chat_message(USER_ID, "assistant", response)

        print("Assistant:", response, "\n")


if __name__ == "__main__":
    run_cli()
