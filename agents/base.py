# agents/base.py
# Shared helpers for all agents.
# DEMO MODE: If OPENAI_API_KEY is not set, returns a realistic pre-written analysis
# so the full graph can be demonstrated without any API key.

import os
from config.settings import care_config

DEMO_MODE = not bool(care_config.openai_api_key)

if DEMO_MODE:
    print("\n  [INFO] No OPENAI_API_KEY found — running in DEMO MODE.")
    print("  [INFO] All agent analyses will use pre-written clinical text.")
    print("  [INFO] Set OPENAI_API_KEY in .env to enable live LLM reasoning.\n")


def call_llm(system_prompt: str, user_prompt: str, demo_response: str) -> str:
    """
    Call the LLM with system + user prompt.
    Falls back to demo_response if no API key is configured.
    """
    if DEMO_MODE:
        return demo_response

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOpenAI(
            model=care_config.model_name,
            temperature=care_config.temperature,
            api_key=care_config.openai_api_key,
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        return response.content

    except Exception as e:
        return f"[LLM ERROR: {e}]\n\n{demo_response}"


def call_llm_with_tools(system_prompt: str, user_prompt: str,
                         tools: list, demo_tool_calls: list) -> list:
    """
    Run an LLM agentic loop with tool use.
    In demo mode, returns pre-defined tool calls directly without LLM.
    Returns a list of (tool_name, tool_args) tuples executed.
    """
    if DEMO_MODE:
        return demo_tool_calls

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        import json

        llm = ChatOpenAI(
            model=care_config.model_name,
            temperature=care_config.temperature,
            api_key=care_config.openai_api_key,
        )
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        executed = []
        for _ in range(3):   # max 3 agentic rounds
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                break

            for tc in response.tool_calls:
                result = tool_map[tc["name"]].invoke(tc["args"])
                executed.append((tc["name"], tc["args"]))
                messages.append(ToolMessage(
                    content=str(result), tool_call_id=tc["id"]
                ))

        return executed

    except Exception as e:
        print(f"  [LLM tool-call error: {e}] — using demo tool calls")
        return demo_tool_calls

