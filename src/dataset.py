import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from bio_features import calculate_physicochemical_props


class HybridProteinDataset(Dataset):
    def __init__(
        self, data_path, tokenizer, max_length=1022, label_encoder=None, scaler=None
    ):
        """
        Dataset que prepara tanto la secuencia (para ESM-2) como
        las características numéricas (para el MLP).
        """
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 1. Procesar Etiquetas (Multi-Label)
        # Asumimos formato de lista o convertimos string a lista
        self.df["labels_list"] = self.df["location_label"].apply(
            lambda x: eval(x) if isinstance(x, str) and "[" in x else [x]
        )

        if label_encoder is None:
            self.mlb = MultiLabelBinarizer()
            self.labels = self.mlb.fit_transform(self.df["labels_list"])
        else:
            self.mlb = label_encoder
            self.labels = self.mlb.transform(self.df["labels_list"])

        # 2. Generar/Cargar Características Biológicas Numéricas
        # Calculamos on-the-fly si no existen columnas 'bio_'
        if not any(col.startswith("bio_") for col in self.df.columns):
            print("Calculando características biofísicas...")
            bio_props = (
                self.df["sequence"]
                .apply(calculate_physicochemical_props)
                .apply(pd.Series)
            )
            self.bio_features = bio_props.values
        else:
            bio_cols = [c for c in self.df.columns if c.startswith("bio_")]
            self.bio_features = self.df[bio_cols].values

        # 3. Normalización (StandardScaler) - CRUCIAL para redes neuronales
        if scaler is None:
            self.scaler = StandardScaler()
            self.bio_features = self.scaler.fit_transform(self.bio_features)
        else:
            self.scaler = scaler
            self.bio_features = self.scaler.transform(self.bio_features)

        # Convertir a tensores float32
        self.bio_features = torch.tensor(self.bio_features, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # A. Datos para ESM-2
        sequence = self.df.iloc[idx]["sequence"]
        encoding = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        # B. Datos Numéricos Extra
        item["extra_features"] = self.bio_features[idx]

        # C. Etiquetas
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item
