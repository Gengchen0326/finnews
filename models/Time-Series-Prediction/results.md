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
        "batch_size": 64,
        "learning_rate": 0.0001,
        "train_test_split": 0.6,
        "epochs": 20,
        "data_path": "data_5y",
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
        "input_dim": 6,
        "hidden_dim": 64,
        "output_dim": 1,
        "num_layers": 2,
        "dropout": 0.2,
        "sequence_length": 20,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "train_test_split": 0.6,
        "epochs": 20,
        "data_path": "data_5y",
        "use_combined_dataloader": true,
        "columns": ["Close", "Volume", "Sentiment_vader","Positive","Negative"],
        "columns_to_normalise": [1, 2, 3, 4, 5],
        "normalise": true,
        "train": true,
        "train_from_scratch": true,
        "predict": true
    }
} 

Nonsentiment:
Mean Squared Error (MSE): 0.00078016
Root Mean Squared Error (RMSE): 0.02793131
Mean Absolute Error (MAE): 0.01723194
Directional Accuracy: 0.5183
R-squared (R2): -0.0011

Sentiment:
Mean Squared Error (MSE): 0.00063950
Root Mean Squared Error (RMSE): 0.02528827
Mean Absolute Error (MAE): 0.01614084
Directional Accuracy: 0.5227
R-squared (R2): 0.0073

# LSTM 配置与gru完全相同
Nonsentiment:
Mean Squared Error (MSE): 0.00077985
Root Mean Squared Error (RMSE): 0.02792585
Mean Absolute Error (MAE): 0.01722657
Directional Accuracy: 0.5194
R-squared (R2): -0.0007

Sentiment:
Mean Squared Error (MSE): 0.00063805
Root Mean Squared Error (RMSE): 0.02525973
Mean Absolute Error (MAE): 0.01612637
Directional Accuracy: 0.5178
R-squared (R2): 0.0096
Metrics saved to results/sentiment_model/metrics_2025051421.csv

# transformer 增加了两个配置- Transformer模型维度: 512 - 注意力头数: 8 更改了两处lr=0.00005 epoch=10


sentiment
Mean Squared Error (MSE): 0.00065034
Root Mean Squared Error (RMSE): 0.02550175
Mean Absolute Error (MAE): 0.01634897
Directional Accuracy: 0.4693
R-squared (R2): -0.0095


    "sentiment_model": {
        "model_type": "transformer",
        "input_dim": 6,
        "hidden_dim": 64,
        "output_dim": 1,
        "num_layers": 2,
        "dropout": 0.2,
        "sequence_length": 20,
        "batch_size": 8,
        "learning_rate": 0.00005,
        "train_test_split": 0.6,
        "epochs": 15,
        "data_path": "data_5y",
        "use_combined_dataloader": true,
        "columns": ["Close", "Volume", "Sentiment_vader","Positive","Negative"],
        "columns_to_normalise": [1, 2, 3, 4, 5],
        "normalise": true,
        "train": true,
        "train_from_scratch": true,
        "predict": true
    }
Mean Squared Error (MSE): 0.00064055
Root Mean Squared Error (RMSE): 0.02530904
Mean Absolute Error (MAE): 0.01611969
Directional Accuracy: 0.5244
R-squared (R2): 0.0057