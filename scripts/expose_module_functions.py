import ast
import os


def extract_functions(module_path):
    """Extracts function names from a Python file, ignoring private functions (_prefix)."""
    with open(module_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=module_path)

    return [node.name for node in tree.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")]


def generate_util_init(package_path="pysi/util"):
    init_path = os.path.join(package_path, "__init__.py")
    modules = [f[:-3] for f in os.listdir(package_path) if f.endswith(".py") and f != "__init__.py"]

    all_functions = []
    lines = []

    for module in modules:
        module_path = os.path.join(package_path, f"{module}.py")
        functions = extract_functions(module_path)

        if functions:
            lines.append(f"from .{module} import {', '.join(functions)}")
            all_functions.extend(functions)

    with open(init_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
        f.write(f"__all__ = {all_functions}\n")

    print(f"Generated {init_path}")


# Run the function
generate_util_init()
