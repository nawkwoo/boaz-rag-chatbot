### utils.py

def truncate_sparse_vector(sv, k=800):
    if not sv or "indices" not in sv or len(sv["indices"]) == 0:
        return {"indices": [], "values": []}
    if len(sv["indices"]) <= k:
        return sv
    topk = sorted(
        zip(sv["indices"], sv["values"]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:k]
    return {
        "indices": [i for i, _ in topk],
        "values": [v for _, v in topk]
    }
