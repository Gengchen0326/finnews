# GRU
{
    "nonsentiment_model": {
        "model_type": "gru",
        "input_dim": 3,
        "hidden_dim": 64,
        "output_dim": 1,
        "num_layers": 2,
        "dropout": 0.2,
        "sequence_length": 20,
        "batch_size": 16,
        "learning_rate": 0.0005,
        "train_test_split": 0.6,
        "epochs": 20,
        "data_path": "data_1y",
        "use_combined_dataloader": true,
        "columns": ["Close", "Volume"],
        "columns_to_normalise": [1, 2],
        "normalise": true,
        "train": true,
        "train_from_scratch": true,
        "predict": true
    },
    "sentiment_model": {
        "model_type": "gru",
        "input_dim": 4,
        "hidden_dim": 64,
        "output_dim": 1,
        "num_layers": 2,
        "dropout": 0.2,
        "sequence_length": 20,
        "batch_size": 16,
        "learning_rate": 0.0005,
        "train_test_split": 0.6,
        "epochs": 20,
        "data_path": "data_1y",
        "use_combined_dataloader": true,
        "columns": ["Close", "Volume", "Sentiment_vader"],
        "columns_to_normalise": [1, 2, 3],
        "normalise": true,
        "train": true,
        "train_from_scratch": true,
        "predict": true
    }
} 

Mean Squared Error (MSE): 0.00031633
Root Mean Squared Error (RMSE): 0.01778570
Mean Absolute Error (MAE): 0.01247208
Directional Accuracy: 0.5207
R-squared (R2): 0.0005

Mean Squared Error (MSE): 0.00031634
Root Mean Squared Error (RMSE): 0.01778604
Mean Absolute Error (MAE): 0.01247501
Directional Accuracy: 0.5206
R-squared (R2): 0.0004

# LSTM
Mean Squared Error (MSE): 0.00031648
Root Mean Squared Error (RMSE): 0.01778994
Mean Absolute Error (MAE): 0.01247889
Directional Accuracy: 0.5207
R-squared (R2): -0.0000

Mean Squared Error (MSE): 0.00031636
Root Mean Squared Error (RMSE): 0.01778647
Mean Absolute Error (MAE): 0.01247401
Directional Accuracy: 0.5204
R-squared (R2): 0.0004

# Transformer

{
    "nonsentiment_model": {
        "model_type": "transformer",
        "input_dim": 3,
        "hidden_dim": 64,
        "output_dim": 1,
        "num_layers": 2,
        "dropout": 0.2,
        "sequence_length": 20,
        "batch_size": 16,
        "learning_rate": 0.00005,
        "train_test_split": 0.6,
        "epochs": 20,
        "data_path": "data_1y_augment",
        "use_combined_dataloader": true,
        "columns": ["Close", "Volume"],
        "columns_to_normalise": [1, 2],
        "normalise": true,
        "train": true,
        "train_from_scratch": true,
        "predict": true
    },
    "sentiment_model": {
        "model_type": "transformer",
        "input_dim": 4,
        "hidden_dim": 64,
        "output_dim": 1,
        "num_layers": 2,
        "dropout": 0.2,
        "sequence_length": 20,
        "batch_size": 16,
        "learning_rate": 0.00005,
        "train_test_split": 0.6,
        "epochs": 20,
        "data_path": "data_1y_augment",
        "use_combined_dataloader": true,
        "columns": ["Close", "Volume", "new_sentiment"],
        "columns_to_normalise": [1, 2, 3, 4, 5],
        "normalise": true,
        "train": true,
        "train_from_scratch": true,
        "predict": true
    }
} 

Mean Squared Error (MSE): 0.00031649
Root Mean Squared Error (RMSE): 0.01779023
Mean Absolute Error (MAE): 0.01247630
Directional Accuracy: 0.5207
R-squared (R2): -0.0000


Mean Squared Error (MSE): 0.00031618
Root Mean Squared Error (RMSE): 0.01778135
Mean Absolute Error (MAE): 0.01246943
Directional Accuracy: 0.5173
R-squared (R2): 0.0009