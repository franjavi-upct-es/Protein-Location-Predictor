import torch
import torch.nn as nn
from transformers import EsmModel, EsmPreTrainedModel


class ESM2WithBioFeatures(EsmPreTrainedModel):
    def __init__(self, config, num_extra_features):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 1. Cuerpo del modelo ESM-2 (Pre-entrenado)
        self.esm = EsmModel(config)

        # 2. Capa de proyección para las características extra (para darles peso)
        self.bio_feature_projector = nn.Sequential(
            nn.Linear(num_extra_features, 32), nn.ReLU(), nn.Dropout(0.1)
        )

        # 3. Clasificador Final
        # Entrada: Tamaño del embedding de ESM + Tamaño de características bio procesadas
        final_input_size = config.hidden_size + 32

        self.classifier = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(final_input_size, config.num_labels)
        )

        # Inicializar pesos (importante para la convergencia)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        extra_features=None,
        labels=None,
        **kwargs,
    ):
        # A. Paso por ESM-2
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Usamos el token CLS (el primero) como representación de la secuencia
        esm_embedding = sequence_output[:, 0, :]

        # B. Procesar características extra
        # extra_features viene del Dataset: [batch_size, num_bio_features]
        bio_embedding = self.bio_feature_projector(extra_features)

        # C. Concatenación
        combined_vector = torch.cat((esm_embedding, bio_embedding), dim=1)

        # D. Clasificación
        logits = self.classifier(combined_vector)

        loss = None
        if labels is not None:
            # Usamos BCEWithLogitsLoss para Multi-Label
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return (
            {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
        )
