"""Generate claim predictions and compare with actuals, saving results to CSV."""
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

from jepa_utils.config import Config
from jepa_utils.data_prep import prepare_data
from jepa_utils.prediction_utils import decode_predicted_codes
from jepa_models.hierarchical_model import HierarchicalClaimsModel


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate model and write predictions to CSV")
    parser.add_argument("checkpoint", help="Path to trained checkpoint")
    parser.add_argument("--output", default="predictions.csv", help="CSV file to write")
    args = parser.parse_args(argv)

    config = Config()
    _, _, eval_dataset, _, config, dataset = prepare_data(config)

    model = HierarchicalClaimsModel.load_from_checkpoint(args.checkpoint, config=config, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(eval_dataset, batch_size=128, collate_fn=dataset.collate_fn, shuffle=False)

    results = []
    with torch.no_grad():
        for batch in dataloader:
            cpt_tensor, icd_tensor, ttnc_tensor, target = batch
            cpt_tensor = cpt_tensor.to(device)
            icd_tensor = icd_tensor.to(device)
            ttnc_tensor = ttnc_tensor.to(device)
            target = target.to(device)

            outputs = model(
                cpt_tensor=cpt_tensor,
                icd_tensor=icd_tensor,
                ttnc_tensor=ttnc_tensor,
                generation=True,
                teacher_forcing=False,
            )

            predicted_cpt_codes = outputs["predicted_cpt_codes"]
            predicted_icd_codes = outputs["predicted_icd_codes"]
            predicted_ttnc_code = outputs["predicted_ttnc_code"]

            decoded_cpt = decode_predicted_codes(predicted_cpt_codes, config.cpt_id_to_token)
            decoded_icd = decode_predicted_codes(predicted_icd_codes, config.icd_id_to_token)

            for i in range(cpt_tensor.size(0)):
                predicted_cpt_list = decoded_cpt[i]
                predicted_icd_list = decoded_icd[i]
                actual_cpt_indices = cpt_tensor[i, -1, :].tolist()
                actual_cpt = [config.cpt_id_to_token.get(idx, "<UNK>") for idx in actual_cpt_indices if idx != 0]
                actual_icd_indices = icd_tensor[i, -1, :].tolist()
                actual_icd = [config.icd_id_to_token.get(idx, "<UNK>") for idx in actual_icd_indices if idx != 0]
                predicted_ttnc_idx = predicted_ttnc_code[i].item()
                predicted_ttnc = config.ttnc_id_to_token.get(predicted_ttnc_idx, "<UNK>")
                actual_ttnc_idx = ttnc_tensor[i, -1].item()
                actual_ttnc = config.ttnc_id_to_token.get(actual_ttnc_idx, "<UNK>") if actual_ttnc_idx != 0 else ""

                results.append(
                    {
                        "predicted_cpt": " ".join(predicted_cpt_list),
                        "actual_cpt": " ".join(actual_cpt),
                        "predicted_icd": " ".join(predicted_icd_list),
                        "actual_icd": " ".join(actual_icd),
                        "predicted_ttnc": predicted_ttnc,
                        "actual_ttnc": actual_ttnc,
                        "target": target[i].item(),
                    }
                )

    pd.DataFrame(results).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
