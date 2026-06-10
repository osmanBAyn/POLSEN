Chemical Recyclability (ROP Enthalpy)
| Model            | Hyperparameters                                                                                                                                       |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Random Forest    | `n_estimators=400`, `max_depth=20`                                                                                                                    |
| XGBoost          | `n_estimators=700`, `learning_rate=0.03`, `max_depth=6`                                                                                               |
| LightGBM         | `n_estimators=1200`, `learning_rate=0.02`, `num_leaves=24`, `max_depth=7`, `subsample=0.8`, `colsample_bytree=0.7`, `reg_alpha=0.1`, `reg_lambda=1.0` |
| MLP              | `hidden_layer_sizes=(128,64)`, `max_iter=1000`                                                                                                        |
| KNN              | `n_neighbors=5`                                                                                                                                       |
| Ridge Regression | `alpha=1.0`                                                                                                                                           |
Degradability
| Model            | Hyperparameters                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| Random Forest    | `n_estimators=200`                                                                               |
| XGBoost          | `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8` |
| LightGBM         | `n_estimators=1400`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.7`               |
| MLP              | `hidden_layer_sizes=(200,100)`, `max_iter=500`                                                   |
| KNN              | `n_neighbors=5`                                                                                  |
| Ridge Regression | `alpha=1.0`                                                                                      |
Hansen Solubility Parameter
| Model            | Hyperparameters                                                                                                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Random Forest    | `n_estimators=500`                                                                                                                                                                                |
| XGBoost          | `n_estimators=1155`, `learning_rate=0.07`, `max_depth=10`, `subsample=1.0`, `colsample_bytree=1.0`, `min_child_weight=1`, `gamma=3.18e-07` *(optimized using Optuna and 5-fold cross-validation)* |
| LightGBM         | `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`                                                                                                                          |
| MLP              | `hidden_layer_sizes=(256,128,64)`, `max_iter=500`                                                                                                                                                 |
| KNN              | `n_neighbors=5`                                                                                                                                                                                   |
| Ridge Regression | `alpha=1.0`                                                                                                                                                                                       |
Melting Temperature (Tm)
| Model            | Hyperparameters                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| Random Forest    | `n_estimators=100`, `max_depth=15`                                                                |
| XGBoost          | `n_estimators=1500`, `max_depth=8`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8` |
| LightGBM         | `n_estimators=1500`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`                |
| MLP              | `hidden_layer_sizes=(100,50)`, `max_iter=500`                                                     |
| KNN              | `n_neighbors=5`                                                                                   |
| Ridge Regression | `alpha=1.0`                                                                                       |
Glass Transition Temperature (Tg)
| Model            | Hyperparameters                                                                                                                                                       |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Random Forest    | `n_estimators=100`                                                                                                                                                    |
| XGBoost          | `n_estimators=750`, `learning_rate=0.05`, `max_depth=10`, `subsample=0.8`, `colsample_bytree=0.7`, `gamma=5`, `reg_alpha=7.5`, `min_child_weight=6`, `reg_lambda=2.3` |
| LightGBM         | `n_estimators=100`                                                                                                                                                    |
| MLP              | `hidden_layer_sizes=(100,50)`, `max_iter=500`                                                                                                                         |
| KNN              | `n_neighbors=5`                                                                                                                                                       |
| Ridge Regression | `alpha=1.0`                                                                                                                                                           |
Decomposition Temperature (Td)
| Model            | Hyperparameters                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| Random Forest    | `n_estimators=100`, `max_depth=15`                                                                |
| XGBoost          | `n_estimators=1500`, `max_depth=8`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8` |
| LightGBM         | `n_estimators=1500`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`                |
| MLP              | `hidden_layer_sizes=(100,50)`, `max_iter=500`                                                     |
| KNN              | `n_neighbors=5`                                                                                   |
| Ridge Regression | `alpha=1.0`                                                                                       |
Refractive Index
| Model            | Hyperparameters                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| Random Forest    | `n_estimators=530`, `max_depth=20`, `min_samples_split=9`, `min_samples_leaf=1`, `max_features=1` |
| XGBoost          | `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`                                           |
| LightGBM         | `n_estimators=500`, `learning_rate=0.05`                                                          |
| MLP              | `hidden_layer_sizes=(200,100)`, `max_iter=1000`                                                   |
| KNN              | `n_neighbors=5`                                                                                   |
| Ridge Regression | `alpha=1.0`                                                                                       |
Limiting Oxygen Index (LOI)
| Model            | Hyperparameters                                         |
| ---------------- | ------------------------------------------------------- |
| Random Forest    | `n_estimators=200`                                      |
| XGBoost          | `n_estimators=500`, `max_depth=6`, `learning_rate=0.05` |
| LightGBM         | `n_estimators=500`, `learning_rate=0.05`                |
| MLP              | `hidden_layer_sizes=(200,100)`, `max_iter=1000`         |
| KNN              | `n_neighbors=5`                                         |
| Ridge Regression | `alpha=1.0`                                             |
Thermal Conductivity and Coefficient of Thermal Expansion (CTE)
| Model            | Hyperparameters                                         |
| ---------------- | ------------------------------------------------------- |
| Random Forest    | `n_estimators=200`                                      |
| XGBoost          | `n_estimators=500`, `max_depth=6`, `learning_rate=0.05` |
| LightGBM         | `n_estimators=500`, `learning_rate=0.05`                |
| MLP              | `hidden_layer_sizes=(200,100)`, `max_iter=1000`         |
| KNN              | `n_neighbors=5`                                         |
| Ridge Regression | `alpha=1.0`                                             |
Gas Permeability
| Model            | Hyperparameters                                          |
| ---------------- | -------------------------------------------------------- |
| Random Forest    | `n_estimators=100`                                       |
| XGBoost          | `n_estimators=1000`, `max_depth=6`, `learning_rate=0.05` |
| LightGBM         | `n_estimators=1000`, `learning_rate=0.05`                |
| MLP              | `hidden_layer_sizes=(200,100)`, `max_iter=1000`          |
| KNN              | `n_neighbors=10`                                         |
| Ridge Regression | `alpha=1.0`                                              |
Dielectric Constant (EPS)
| Model            | Hyperparameters                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| Random Forest    | `n_estimators=200`                                                                                |
| XGBoost          | `n_estimators=1000`, `max_depth=6`, `learning_rate=0.05`, `colsample_bytree=0.7`, `subsample=0.8` |
| LightGBM         | `n_estimators=500`, `learning_rate=0.05`                                                          |
| MLP              | `hidden_layer_sizes=(200,100)`, `max_iter=2000`                                                   |
| KNN              | `n_neighbors=7`, `weights='distance'`                                                             |
| Ridge Regression | `alpha=1.0`                                                                                       |
Hildebrand Solubility Parameter
| Model            | Hyperparameters                                         |
| ---------------- | ------------------------------------------------------- |
| Random Forest    | `n_estimators=200`                                      |
| XGBoost          | `n_estimators=500`, `max_depth=6`, `learning_rate=0.05` |
| LightGBM         | `n_estimators=500`, `learning_rate=0.05`                |
| MLP              | `hidden_layer_sizes=(200,100)`, `max_iter=1000`         |
| KNN              | `n_neighbors=5`                                         |
| Ridge Regression | `alpha=1.0`                                             |
Bandgap Bulk
| Model            | Hyperparameters                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| Random Forest    | `n_estimators=100`                                                                               |
| XGBoost          | `n_estimators=1000`, `learning_rate=0.2`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8` |
| LightGBM         | `n_estimators=100`                                                                               |
| MLP              | `hidden_layer_sizes=(100,50)`, `max_iter=500`                                                    |
| KNN              | `n_neighbors=5`                                                                                  |
| Ridge Regression | `alpha=1.0`                                                                                      |
Bandgap Chain and Bandgap Crystal
| Model            | Hyperparameters                                                                                                             |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Random Forest    | `n_estimators=100`                                                                                                          |
| XGBoost          | `n_estimators=100`                                                                                                          |
| LightGBM         | `n_estimators=1500`, `learning_rate=0.03`, `num_leaves=40`, `min_child_samples=10`, `subsample=0.8`, `colsample_bytree=0.7` |
| MLP              | `hidden_layer_sizes=(100,50)`, `max_iter=500`                                                                               |
| KNN              | `n_neighbors=5`                                                                                                             |
| Ridge Regression | `alpha=1.0`                                                                                                                 |
