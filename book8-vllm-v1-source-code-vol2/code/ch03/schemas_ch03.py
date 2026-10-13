# schemas_ch03.py —— ch3 实验 schema 变体集（固定种子、确定性；exp1/exp3 共用）
# 所属章节：Book8 ch3（结构化输出）；设计书=notes/12 §四（8 变体/4 正确性变体）
# 运行：python code/Book8-vLLM源码剖析下/ch03/schemas_ch03.py   （打印变体清单；被 exp1/exp3 import）
VARIANTS = {
    "obj_basic": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": False,
    },
    "obj_nested": {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {"id": {"type": "integer"},
                               "tags": {"type": "array", "items": {"type": "string"}}},
                "required": ["id"],
            },
            "active": {"type": "boolean"},
        },
        "required": ["user"],
    },
    "arr_basic": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6},
    "enum_basic": {
        "type": "object",
        "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
        "required": ["color"],
        "additionalProperties": False,
    },
    "obj_many_props": {
        "type": "object",
        "properties": {f"field_{i}": {"type": "integer"} for i in range(12)},
        "required": [f"field_{i}" for i in range(6)],
    },
    "str_pattern": {
        "type": "object",
        "properties": {"code": {"type": "string", "pattern": "^[A-Z]{3}-[0-9]{4}$"}},
        "required": ["code"],
    },
    "num_range": {
        "type": "object",
        "properties": {"score": {"type": "number", "minimum": 0, "maximum": 100}},
        "required": ["score"],
    },
    "deep_nested": {
        "type": "object",
        "properties": {
            "a": {
                "type": "object",
                "properties": {
                    "b": {
                        "type": "object",
                        "properties": {
                            "c": {"type": "array", "items": {"type": "integer"}}
                        },
                        "required": ["c"],
                    }
                },
                "required": ["b"],
            }
        },
        "required": ["a"],
    },
}
PROMPTS = {
    "obj_basic": "Generate a person profile with a name and an age.",
    "obj_nested": "Generate a user record with an id and some tags, and an active flag.",
    "arr_basic": "List between 2 and 6 fruit names.",
    "enum_basic": "Pick one color from red, green, blue.",
    "obj_many_props": "Fill twelve integer fields field_0 to field_11 with small numbers.",
    "str_pattern": "Produce a code like ABC-1234 (three uppercase letters, dash, four digits).",
    "num_range": "Give a score between 0 and 100.",
    "deep_nested": "Produce a nested object a.b.c where c is a list of integers.",
}
CORRECTNESS_SET = ["obj_basic", "obj_nested", "arr_basic", "enum_basic"]   # exp3 的 4 变体×8 请求

if __name__ == "__main__":
    for k, v in VARIANTS.items():
        print(f"{k:16s} {str(v)[:90]}")
    print("correctness set:", CORRECTNESS_SET)
