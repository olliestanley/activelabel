from typing import Any


class CachedFunctionKV:
    def __init__(self, function: callable):
        self.function = function
        self.cache = {}

    def __call__(self, key) -> Any:
        if key not in self.cache:
            self.cache[key] = self.function(key)
        return self.cache[key]
