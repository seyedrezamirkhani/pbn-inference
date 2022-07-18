from ctypes import util
from pathlib import Path

from pbn_inference.utils import spawn

if __name__ == "__main__":
    predictor_sets_path = Path(__file__).parent / "pbn_inference" / "data"
    genedata = predictor_sets_path / "genedata.xls"
    TOTAL_GENES = 200

    includeIDs = [234237, 324901, 759948, 25485, 266361, 108208, 130057]
    graph = spawn(
        file=genedata,
        total_genes=TOTAL_GENES,
        include_ids=includeIDs,
        bin_method="kmeans",
        n_predictors=5,
        predictor_sets_path=predictor_sets_path,
    )

    graph.genRandState()
    print(graph.getState())
    for _ in range(10):
        graph.step()
        print(graph.getState())
