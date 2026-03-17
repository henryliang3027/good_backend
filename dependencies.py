"""
共用依賴注入。
在 lifespan 初始化完成後呼叫 set_collection() 注入 ChromaDB collection，
各 router 透過 Depends(get_collection) 取用。
"""

_collection = None


def set_collection(col):
    global _collection
    _collection = col


def get_collection():
    return _collection
