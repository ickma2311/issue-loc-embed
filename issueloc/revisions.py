"""Pinned Hugging Face revisions used for every reported number (none of these repositories had a later commit
when the runs were made). Datasets are read from their auto-converted parquet branch at the given commit."""

DATASET_PARQUET_REV = {
    "nebius/SWE-rebench": "21fe2abc5628fee15fe93d211dcd303ca85f3683",
    "nebius/SWE-rebench-V2": "4d10ee08ee246f5d7475c3cef8baea030fdf4942",
    "czlll/SWE-bench_Lite": "e2bff55659263a2eae404e38eb821bd1c11e4cb4",
    "princeton-nlp/SWE-bench": "9d0f6fa0295e7b65123111da98d53ed6c9bec66d",
    "princeton-nlp/SWE-bench_Lite": "177b1fa24f05b6351ad8e4439d26ac68b7ba27ff",
    "czlll/Loc-Bench_V1": "f3087a02d989bd31beef561321cdfcf0a450f826",
    "czlll/Loc-Bench_V0.2": "cb5f2a2fd17981d9b2cfece8d7327f5989b7091b",
    "czlll/Loc-Bench_V0.1": "2fb1064f5bd99f1b9d77e3a43fad51de2524ddab",
    "mteb/CodeSearchNetRetrieval": "698f8e1f6f9afc8c3df12543f5f1b203b3af5842",
}

# The LocBench corpus (build_locbench.py) reads the main-branch file at this commit (same data as the parquet branch).
LOCBENCH_MAIN_REV = "c44cf3b74e07ca642cec841b471a9939907c12a7"

MODEL_REV = {
    "nomic-ai/CodeRankEmbed": "3c4b60807d71f79b43f3c4363786d9493691f8b1",
    "Salesforce/SweRankEmbed-Small": "745d2a06103a66d3cfa600aa52fc0d3523010daa",
    "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "lightonai/LateOn-Code": "ace431824f35db231178fc602e33296784762a2e",
    "lightonai/LateOn-Code-edge": "4bcdf5ed93f791259eb130b577a240f753d68dd8",
}


def model_rev(name: str) -> str | None:
    """Pinned revision for a hub model id; None for local directories."""
    return MODEL_REV.get(name)


def parquet_url(ds: str, config: str, split: str, part: str) -> str:
    return f"https://huggingface.co/datasets/{ds}/resolve/{DATASET_PARQUET_REV[ds]}/{config}/{split}/{part}.parquet"
