import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from jepa_utils.config import Config
from jepa_utils.data_prep import prepare_data
from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.prediction_utils import decode_predicted_codes


def generate_predictions(ckpt_path: str) -> None:
    cfg = Config()
    # ensure dataset split for evaluation
    cfg.use_generative_save = True
    cfg.use_lr_find = False
    cfg.use_plotting = False

    _, _, eval_dataset, _, cfg, dataset = prepare_data(cfg)

    model = HierarchicalClaimsModel.load_from_checkpoint(ckpt_path, config=cfg, strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        eval_dataset,
        batch_size=128,
        collate_fn=dataset.collate_fn,
        shuffle=False
    )

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            cpt_tensor, icd_tensor, ttnc_tensor, target = [x.to(device) for x in batch]

            outputs = model(
                cpt_tensor=cpt_tensor,
                icd_tensor=icd_tensor,
                ttnc_tensor=ttnc_tensor,
                generation=True,
                teacher_forcing=False
            )

            decoded_cpt = decode_predicted_codes(outputs['predicted_cpt_codes'], cfg.cpt_id_to_token)
            decoded_icd = decode_predicted_codes(outputs['predicted_icd_codes'], cfg.icd_id_to_token)

            for i in range(cpt_tensor.size(0)):
                actual_cpt_indices = cpt_tensor[i, -1, :].cpu().numpy()
                actual_cpt = [cfg.cpt_id_to_token.get(idx, '<UNK>') for idx in actual_cpt_indices if idx != 0]

                actual_icd_indices = icd_tensor[i, -1, :].cpu().numpy()
                actual_icd = [cfg.icd_id_to_token.get(idx, '<UNK>') for idx in actual_icd_indices if idx != 0]

                predicted_ttnc_idx = outputs['predicted_ttnc_code'][i].item()
                predicted_ttnc = cfg.ttnc_id_to_token.get(predicted_ttnc_idx, '<UNK>')

                actual_ttnc_idx = ttnc_tensor[i, -1].item()
                actual_ttnc = cfg.ttnc_id_to_token.get(actual_ttnc_idx, '<UNK>') if actual_ttnc_idx != 0 else ''

                results.append({
                    'predicted_cpt': ' '.join(decoded_cpt[i]),
                    'actual_cpt': ' '.join(actual_cpt),
                    'predicted_icd': ' '.join(decoded_icd[i]),
                    'actual_icd': ' '.join(actual_icd),
                    'predicted_ttnc': predicted_ttnc,
                    'actual_ttnc': actual_ttnc,
                    'target': target[i].item()
                })

    pd.DataFrame(results).to_csv('predictions.csv', index=False)
    print('Saved predictions to predictions.csv')


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate claims predictions")
    parser.add_argument('ckpt', help='Path to checkpoint file')
    args = parser.parse_args()
    generate_predictions(args.ckpt)


if __name__ == '__main__':
    main()
