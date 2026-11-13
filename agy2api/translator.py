import json
from typing import Any, Dict, List, Optional

INTEGER_SCHEMA_KEYS = {
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minProperties",
    "maxProperties",
}

NUMBER_SCHEMA_KEYS = {
    "minimum",
    "maximum",
    "multipleOf",
}

def normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return schema

    new_schema = dict(schema)
    if "type" in new_schema and isinstance(new_schema["type"], str):
        new_schema["type"] = new_schema["type"].lower()

    for key in INTEGER_SCHEMA_KEYS:
        if key in new_schema:
            val = new_schema[key]
            if isinstance(val, str):
                try:
                    new_schema[key] = int(val)
                except (ValueError, TypeError):
                    del new_schema[key]

    for key in NUMBER_SCHEMA_KEYS:
        if key in new_schema:
            val = new_schema[key]
            if isinstance(val, str):
                try:
                    new_schema[key] = float(val) if "." in val else int(val)
                except (ValueError, TypeError):
                    del new_schema[key]

    if "properties" in new_schema and isinstance(new_schema["properties"], dict):
        new_props = {}
        for prop_name, prop_def in new_schema["properties"].items():
            new_props[prop_name] = normalize_schema(prop_def)
        new_schema["properties"] = new_props

    if "items" in new_schema:
        if isinstance(new_schema["items"], dict):
            new_schema["items"] = normalize_schema(new_schema["items"])
        elif isinstance(new_schema["items"], list):
            new_schema["items"] = [normalize_schema(i) for i in new_schema["items"]]

    for comb_key in ("anyOf", "allOf", "oneOf"):
        if comb_key in new_schema and isinstance(new_schema[comb_key], list):
            new_schema[comb_key] = [normalize_schema(sub) for sub in new_schema[comb_key]]

    return new_schema

def gemini_to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    openai_tools: List[Dict[str, Any]] = []
    for tool_entry in tools:
        if not isinstance(tool_entry, dict):
            continue
        declarations = tool_entry.get("functionDeclarations") or tool_entry.get("function_declarations") or []
        for decl in declarations:
            name = decl.get("name", "")
            description = decl.get("description", "")
            parameters = normalize_schema(decl.get("parameters", {}))
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            })
    return openai_tools

def extract_text_from_parts(parts: List[Dict[str, Any]]) -> str:
    texts = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            texts.append(part["text"])
    return "".join(texts)

def gemini_to_openai_messages(
    contents: List[Dict[str, Any]],
    system_instruction: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    if system_instruction and "parts" in system_instruction:
        sys_text = extract_text_from_parts(system_instruction["parts"])
        if sys_text:
            messages.append({"role": "system", "content": sys_text})

    from collections import defaultdict
    tool_call_queues: Dict[str, List[str]] = defaultdict(list)
    call_counter = 0

    for item in contents:
        role = item.get("role", "user")
        parts = item.get("parts", [])

        # Check for functionCall or functionResponse
        function_calls = [p["functionCall"] for p in parts if isinstance(p, dict) and "functionCall" in p]
        function_responses = [p["functionResponse"] for p in parts if isinstance(p, dict) and "functionResponse" in p]

        if function_calls:
            tool_calls = []
            for fc in function_calls:
                call_counter += 1
                fn_name = fc.get("name", "function")
                call_id = f"call_{fn_name}_{call_counter}"
                tool_call_queues[fn_name].append(call_id)
                args_dict = fc.get("args", {})
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": json.dumps(args_dict),
                    },
                })
            text = extract_text_from_parts(parts)
            messages.append({
                "role": "assistant",
                "content": text or None,
                "tool_calls": tool_calls,
            })
        elif function_responses or role == "function":
            for fc_resp in function_responses:
                fn_name = fc_resp.get("name", "function")
                if tool_call_queues[fn_name]:
                    call_id = tool_call_queues[fn_name].pop(0)
                else:
                    call_counter += 1
                    call_id = f"call_{fn_name}_{call_counter}"
                resp_data = fc_resp.get("response", {})
                content = json.dumps(resp_data) if isinstance(resp_data, (dict, list)) else str(resp_data)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": fn_name,
                    "content": content,
                })
        else:
            openai_role = "assistant" if role == "model" else "user"
            text = extract_text_from_parts(parts)
            messages.append({"role": openai_role, "content": text})

    return messages

def gemini_to_openai_request(payload: Dict[str, Any], target_model: str) -> Dict[str, Any]:
    clean_model = target_model.removeprefix("models/").split(":")[0]
    contents = payload.get("contents", [])
    system_instruction = payload.get("system_instruction") or payload.get("systemInstruction")
    messages = gemini_to_openai_messages(contents, system_instruction)

    req: Dict[str, Any] = {
        "model": clean_model,
        "messages": messages,
    }

    tools_payload = payload.get("tools")
    if tools_payload:
        tools = gemini_to_openai_tools(tools_payload)
        if tools:
            req["tools"] = tools
            req["tool_choice"] = "auto"

    gen_config = payload.get("generationConfig") or payload.get("generation_config") or {}
    if "temperature" in gen_config:
        req["temperature"] = gen_config["temperature"]
    max_tokens = gen_config.get("maxOutputTokens") or gen_config.get("max_output_tokens")
    if max_tokens:
        req["max_tokens"] = max_tokens
    if "topP" in gen_config or "top_p" in gen_config:
        req["top_p"] = gen_config.get("topP") or gen_config.get("top_p")
    stop_sequences = gen_config.get("stopSequences") or gen_config.get("stop_sequences")
    if stop_sequences:
        req["stop"] = stop_sequences

    return req

def openai_to_gemini_response(openai_data: Dict[str, Any]) -> Dict[str, Any]:
    choices = openai_data.get("choices", [])
    candidates = []

    for choice in choices:
        msg = choice.get("message", {})
        parts: List[Dict[str, Any]] = []

        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning:
            parts.append({"text": reasoning, "thought": True})

        if msg.get("content"):
            parts.append({"text": msg["content"]})

        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}
            parts.append({
                "functionCall": {
                    "name": name,
                    "args": args,
                }
            })

        finish_reason = (choice.get("finish_reason") or "stop").upper()
        if finish_reason == "TOOL_CALLS":
            finish_reason = "STOP"

        candidates.append({
            "content": {
                "role": "model",
                "parts": parts,
            },
            "finishReason": finish_reason,
            "index": choice.get("index", 0),
        })

    usage = openai_data.get("usage", {})
    usage_metadata = {
        "promptTokenCount": usage.get("prompt_tokens", 0),
        "candidatesTokenCount": usage.get("completion_tokens", 0),
        "totalTokenCount": usage.get("total_tokens", 0),
    }

    return {
        "candidates": candidates,
        "usageMetadata": usage_metadata,
    }

class StreamingChunkTranslator:
    def __init__(self):
        self.tool_calls_buffer: Dict[int, Dict[str, Any]] = {}

    def process_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        choices = chunk.get("choices", [])
        if not choices:
            return []

        frames = []
        for choice in choices:
            delta = choice.get("delta", {})
            choice_idx = choice.get("index", 0)
            text_content = delta.get("content")
            reasoning_content = delta.get("reasoning_content") or delta.get("reasoning")
            finish_reason = choice.get("finish_reason")

            # Accumulate tool calls if any
            tool_calls = delta.get("tool_calls", [])
            for tc in tool_calls:
                idx = tc.get("index", 0)
                if idx not in self.tool_calls_buffer:
                    self.tool_calls_buffer[idx] = {
                        "name": "",
                        "arguments": "",
                        "id": tc.get("id", ""),
                    }
                if tc.get("id"):
                    self.tool_calls_buffer[idx]["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    self.tool_calls_buffer[idx]["name"] = fn["name"]
                if fn.get("arguments"):
                    self.tool_calls_buffer[idx]["arguments"] += fn["arguments"]

            # If there is reasoning content
            if reasoning_content:
                frames.append({
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [{"text": reasoning_content, "thought": True}],
                        },
                        "index": choice_idx,
                    }]
                })

            # If there is text content or finish reason without tool calls
            if text_content or (finish_reason and not self.tool_calls_buffer):
                cand: Dict[str, Any] = {
                    "content": {
                        "role": "model",
                        "parts": [{"text": text_content}] if text_content else [],
                    },
                    "index": choice_idx,
                }
                if finish_reason:
                    cand["finishReason"] = finish_reason.upper()
                frames.append({"candidates": [cand]})

            # Flush tool calls when tool call finishes
            if finish_reason and self.tool_calls_buffer:
                flushed = self.flush(finish_reason=finish_reason)
                if flushed:
                    frames.append(flushed)

        return frames


    def flush(self, finish_reason: Optional[str] = "STOP") -> Optional[Dict[str, Any]]:
        if not self.tool_calls_buffer:
            return None

        parts: List[Dict[str, Any]] = []
        for idx in sorted(self.tool_calls_buffer.keys()):
            tc_data = self.tool_calls_buffer[idx]
            args_str = tc_data["arguments"]
            try:
                parsed_args = json.loads(args_str) if args_str else {}
            except Exception:
                parsed_args = {}
            parts.append({
                "functionCall": {
                    "name": tc_data["name"],
                    "args": parsed_args,
                }
            })

        self.tool_calls_buffer.clear()
        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": parts,
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ]
        }

