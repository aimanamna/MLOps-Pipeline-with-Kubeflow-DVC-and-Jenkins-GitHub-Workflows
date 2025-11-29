# pipeline.py
from kfp import dsl
from kfp.v2 import compiler
from kfp.components import load_component_from_file
from pathlib import Path

COMP_DIR = "components"

# If you compiled YAMLs already, load them. If not, you can also import functions directly (kfp supports both).
dvc_get_comp = load_component_from_file(str(Path(COMP_DIR) / "dvc_get_dataset.yaml"))
preprocess_comp = load_component_from_file(str(Path(COMP_DIR) / "preprocess.yaml"))
train_comp = load_component_from_file(str(Path(COMP_DIR) / "train_model.yaml"))
eval_comp = load_component_from_file(str(Path(COMP_DIR) / "evaluate_model.yaml"))


@dsl.pipeline(
    name="mlops-kubeflow-pipeline",
    description="Simple pipeline: fetch dataset via DVC, preprocess, train, evaluate"
)
def ml_pipeline(dvc_repo_url: str = "{{workflow.parameters.dvc_repo_url}}",  # replace default in UI
                raw_out: str = "/tmp/data/raw_data.csv",
                train_npz: str = "/tmp/data/train.npz",
                test_npz: str = "/tmp/data/test.npz",
                model_out: str = "/tmp/model/model.joblib",
                metrics_out: str = "/tmp/output/metrics.json"):
    # 1. Get dataset from DVC remote (dvc get)
    t1 = dvc_get_comp(dvc_remote_url=dvc_repo_url, out_path=raw_out)

    # 2. Preprocess
    t2 = preprocess_comp(input_csv=t1.outputs["return"], output_train=train_npz, output_test=test_npz)

    # 3. Train
    t3 = train_comp(train_npz=t2.outputs["train_file"], model_out=model_out)

    # 4. Evaluate
    t4 = eval_comp(model_file=t3.outputs["return"], test_npz=t2.outputs["test_file"], metrics_out=metrics_out)


if __name__ == "__main__":
    # compile pipeline to pipeline.yaml
    compiler.Compiler().compile(pipeline_func=ml_pipeline, package_path="pipeline.yaml")
    print("Compiled pipeline.yaml")
