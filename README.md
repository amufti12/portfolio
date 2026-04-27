# Aydan Mufti - Data Science & Engineering Portfolio

Hi! I'm Aydan, a data scientist and software engineer with a Master's in Data Science from Boston University (4.0 GPA) and a background in Computer Science from Kennesaw State University. My work spans machine learning, systems programming, and full-stack data pipelines. I'm passionate about building things that are both rigorous and useful - from predictive safety systems for the NFL to real-time sorting algorithm visualizers.

Currently working as a Quality Assurance Engineer at Carrier Global Corporation, where I build PowerBI dashboards and automate testing processes. Actively seeking Data Scientist roles.

aydanmufti@gmail.com &nbsp;|&nbsp; [github.com/amufti12](https://github.com/amufti12)

---

### Contents

1. [NFL Injury Prediction and Risk Analysis System](#nfl-injury-prediction)
2. [HuffPost News Classification (DistilBERT)](#huffpost-classification)
3. [Elden Ring Build Optimizer Agent](#elden-ring-agent)
4. [NHTSA Vehicle Complaints - Azure Cloud Pipeline & Power BI Dashboard](#nhtsa-powerbi)
5. [AlgoViz - Sorting Algorithm Visualizer](#algoviz)
6. [N-Body Gravitational Simulator](#nbody-simulator)
7. [Spotify Song Prediction](#song-prediction)
8. [Video-to-Video Synthesis (C-Day 2021)](#vid2vid)

---

## [NFL Injury Prediction and Risk Analysis System](https://github.com/amufti12/NFL-Injury-Prediction-and-Risk-Analysis-System) <a name="nfl-injury-prediction"></a>

#### Overview
A comprehensive machine learning capstone analyzing NFL player injury patterns across multiple injury mechanisms to identify environmental and contextual risk factors for evidence-based safety policy decisions.

#### Technical Details
- **Datasets**: Three NFL datasets spanning 2016–2020 seasons (surface analytics, punt analytics, impact detection)
- **Models**: Logistic Regression (L1/L2/ElasticNet), Decision Trees, Random Forests, Gradient Boosting, SVMs (linear, RBF, polynomial), KNN, K-Means, DBSCAN, Hierarchical Agglomerative Clustering, Ridge/Lasso/SVR regression
- **Data Pipeline**: Missing value handling (−999 temperature codes), categorical consolidation, polynomial feature engineering (degree 2), stratified 75-25 splits, 3-fold CV, StandardScaler normalization
- **Overfitting Prevention**: Regularization penalties, tree pruning (`max_depth=5`, `min_samples_split=10`), ensemble subsampling, cross-validation gap monitoring
- **Tools**: Python (scikit-learn, pandas, NumPy, matplotlib, seaborn), Jupyter Notebook

#### Model Performance

| Task | Best Model | Accuracy | Notes |
|---|---|---|---|
| Concussion Occurrence (Dataset 2) | Logistic Regression (L2) | **91.7%** | 100% precision |
| Lower Extremity Severity (Dataset 1) | KNN (Euclidean) / SVM (RBF) | **70.0%** | Baseline: 60% |
| Impact Count Prediction (Dataset 3) | SVR (RBF) | MAE: 5.53 | R² = −0.055 |

#### Key Findings
- **Synthetic surfaces increase lower extremity injury rates by ~62%** (rate ratio 1.62, p=0.043) but show **no association with concussions** (rate ratio 1.00, p=1.000) — a mechanism-specific discovery with major policy implications
- Wide receivers, cornerbacks, and linebackers account for 72.7% of lower extremity injuries; returners and gunners show 2.5× elevated concussion risk
- Game context (time, score, field position) accounts for 47% of feature importance in concussion models; temperature interactions drive lower extremity predictions

#### Business Impact
- NFL teams spend ~$521M annually on injured players; preventing 10 injuries offsets significant intervention costs
- Mechanism-specific findings prevent wasteful uniform policies — surface interventions protect lower extremities but won't reduce concussions
- Position-targeted programs address 500 high-risk players vs. 1,696 league-wide

#### Deliverables
- [EDA Notebook](https://github.com/amufti12/NFL-Injury-Prediction-and-Risk-Analysis-System/blob/main/Capstone_EDA_Analysis.ipynb) · [Modeling Notebook](https://github.com/amufti12/NFL-Injury-Prediction-and-Risk-Analysis-System/blob/main/Capstone_Modeling.ipynb)
- Technical Report & Non-Technical Report (APA-formatted) available in repository

---
 
## [HuffPost News Classification (DistilBERT)](https://github.com/amufti12/HuffPost-News-Classification/tree/main) <a name="huffpost-classification"></a>
 
#### Overview
A natural language processing project fine-tuning DistilBERT on HuffPost news headlines to classify articles into topic categories, demonstrating transformer-based text classification on a real-world multi-class problem.
 
#### Technical Details
- **Model**: DistilBERT (distilbert-base-uncased), fine-tuned for multi-class classification
- **Dataset**: HuffPost News Category Dataset (Kaggle)
- **Performance**: 74.4% accuracy, 0.661 macro-F1
- **Tools**: Python (HuggingFace Transformers, PyTorch, pandas, scikit-learn)
 
#### Key Findings
- Transformer-based embeddings significantly outperform classical bag-of-words approaches for headline classification
- Macro-F1 of 0.661 reflects meaningful performance across imbalanced category distribution
- Short headline text presents inherent ambiguity — performance ceiling is constrained by label overlap across categories
 
---

## [Elden Ring Build Optimizer Agent](https://github.com/amufti12/Elden-Ring-Build-Optimizer) <a name="elden-ring-agent"></a>
 
#### Overview
A 10-tool AI agent that recommends complete character builds in Elden Ring (including the Shadow of the Erdtree DLC) by querying a structured 29MB dataset of weapons, upgrades, armor, spells, talismans, and locations — returning data-backed, verifiable recommendations rather than relying on model training knowledge.
 
The agent accepts a natural language build request and autonomously decides which data sources to query, in what order, and how many times — iterating in a ReAct-style loop until it has enough information to produce a complete recommendation covering weapons, affinity, talismans, armor, spirit ash summon, and item locations.
 
#### Technical Details
- **Model**: Gemini (gemini-flash-latest) with function calling
- **Dataset**: [Ultimate Elden Ring with Shadow of the Erdtree DLC](https://www.kaggle.com/datasets/pedroaltobelli/ultimate-elden-ring-with-shadow-of-the-erdtree-dlc) — 15 CSV files, 247 columns, 29MB (CC0 Public Domain)
- **Tools**: Python (pandas, google-genai), Jupyter Notebook
#### Agent Architecture
 
```
User query
    ↓
Gemini with 10 defined tools
    ↓
Model decides which tool(s) to call
    ↓
Tool executes pandas query over cleaned CSV data
    ↓
Result fed back into conversation
    ↓
Repeat until model produces final answer
```
 
The 18MB upgrade table is never loaded into model context — queried on demand via tools, keeping token usage minimal and answers accurate.
 
#### Tools
 
| Tool | Data Source | Purpose |
|---|---|---|
| `query_weapons_by_stat` | `weapons_upgrades.csv` | Filter weapons by scaling grade and stat requirements |
| `get_weapon_upgrades` | `weapons_upgrades.csv` | Compare all affinities at max upgrade for a specific weapon |
| `query_talismans` | `talismans.csv` | Search talismans by keyword in name, effect, or description |
| `query_ashes_of_war` | `ashesOfWar.csv` | Find weapon skills by affinity or skill name |
| `query_spells` | `incantations.csv`, `sorceries.csv` | Search spells by stat requirement and keyword |
| `query_spirit_ashes` | `spiritAshes.csv` | Find summons by FP/HP cost or keyword |
| `query_armor` | `armors.csv` | Search armor by type, weight, or special effect |
| `query_skills` | `skills.csv` | Detailed skill info and equipment compatibility |
| `query_shields` | `shields_upgrades.csv` | Find shields by category or stat scaling |
| `query_locations` | `locations.csv` | Find which areas contain specific items or bosses |
 
#### Key Engineering Decisions
- **Somber weapons fix**: The dataset mislabeled unique weapons (Blasphemous Blade, Rivers of Blood, Moonveil, etc.) as `Standard` through `Standard +10` instead of `Somber +1` through `Somber +10`. The standard max upgrade filter of `Standard +25` silently excluded all of them. Identified the pattern and added `Standard +10` as a valid max upgrade tier alongside `Standard +25`
- **Stringified dict parsing**: Attack power, stat scaling, and requirements were stored as Python dict strings scraped from the wiki. Used `ast.literal_eval` to parse into usable structures before any filtering or computation
- **Scaling grade normalization**: Letter grades (S/A/B/C/D/E) converted to numeric (6/5/4/3/2/1) so weapons can be filtered and sorted by scaling quality rather than just raw attack power
#### Build Types Tested
 
| Build | Primary Stat | Key Result |
|---|---|---|
| Faith/Strength — Holy Knight | 60 Fai / 40 Str | Blasphemous Blade and Maliketh's Black Blade correctly surfaced after somber weapons fix |
| Arcane/Dexterity — Bleed | 60 Arc / 20 Dex | Correctly chose Occult over Blood affinity at 60 Arcane; found Black Whetblade requirement unprompted |
| Intelligence — Pure Sorcerer | 70 Int / 20 Mnd | Recommended Academy Glintstone Staff over Lusat's specifically due to the 20 Mind constraint |
| Dexterity — Samurai/Blademaster | 80 Dex | Found Okina Mask and Curseblade Mask for +Dex helm slot; recommended Black Knife Tiche |
 
---
 
## NHTSA Vehicle Complaints — Azure Cloud Pipeline & Power BI Dashboard <a name="nhtsa-powerbi"></a>
 
#### Overview
An end-to-end cloud data engineering and business intelligence project using NHTSA's public vehicle complaints dataset — a real-world record of consumer-reported automobile issues collected via web and phone. The pipeline spans ingestion, transformation, distributed storage, serverless querying, and interactive visualization across the full Microsoft Azure stack.
 
#### Technical Details
- **Data Source**: NHTSA Complaints Dataset (public record of vehicle complaints across U.S. manufacturers)
- **Cloud Services**: Azure Data Factory (ADF), Azure Data Lake Storage Gen2, Azure Synapse Analytics (Serverless SQL Pool), Azure CosmosDB, Microsoft Power BI
- **Pipeline Architecture**:
  - Raw `.txt` complaint file ingested via ADF and loaded into ADLS Gen2
  - ADF Data Flows used to transform and convert to `.parquet` format, partitioned by manufacturer name
  - Parquet files mounted as an external table in Azure Synapse using PolyBase / serverless SQL
  - A second dataset (TMDB movie data) loaded into CosmosDB via ADF for NoSQL querying
- **SQL Analysis**: Queried Ford F-150 crash records (1990–2010) using Synapse SQL — filtering by model, year range, and crash indicator, then aggregating crash counts by year
- **Power BI Report**:
  - Connected Power BI Desktop to the Synapse serverless SQL endpoint via Azure Synapse Analytics connector
  - Imported NHTSA Synapse external table and NHTSA Ford-specific safety ratings CSV (`Safercar_data_FORD.csv`)
  - Engineered a composite key (`YEAR_MODEL_ID = YEARTXT & "_" & MODELTXT`) to establish a many-to-one relationship between datasets in Model View
  - Built an interactive dashboard with bar/line charts, a data table, and dynamic slicers — all visuals cross-filter on selection
 
#### Key Features
- Full ELT pipeline (Extract → Load → Transform) rather than traditional ETL, reflecting modern cloud data architecture patterns
- Parquet partitioning by manufacturer enables efficient predicate pushdown in distributed queries (simulating production-scale data lake design)
- Cross-dataset joins in Power BI link complaint volume data with safety ratings for richer vehicle-level analysis
- Dashboard slicers allow filtering by model year, vehicle model, and crash indicator dynamically across all visuals
 
#### Tools & Technologies
Azure Data Factory · Azure Data Lake Storage Gen2 · Azure Synapse Analytics · Azure CosmosDB · Microsoft Power BI · SQL · Python · Parquet

![NHTSA Power BI Dashboard](https://github.com/user-attachments/assets/bb8af551-b3fd-4803-9126-59623c587025)
 
[View Interactive Dashboard →](https://app.powerbi.com/view?r=eyJrIjoiMzA3MmZhYzYtYzY0Yy00ZDEyLWI0OGYtMzc0MzEwNjlmYWRkIiwidCI6ImQ1N2QzMmNjLWMxMjEtNDg4Zi1iMDdiLWRmZTcwNTY4MGM3MSIsImMiOjN9)
 
---

## [AlgoViz — Sorting Algorithm Visualizer](https://github.com/amufti12/AlgoViz) <a name="algoviz"></a>

#### Overview
A real-time interactive visualizer for classic sorting algorithms, built in C++ with a custom OpenGL rendering pipeline and an ImGui control panel. Watch Selection, Bubble, Insertion, Merge, and Quick Sort animate step-by-step.

#### Technical Details
- **Language**: C++
- **Rendering**: OpenGL, GLFW, GLM, GLAD
- **UI**: ImGui (immediate-mode GUI)
- **Algorithms Implemented**: Selection Sort, Bubble Sort, Insertion Sort, Merge Sort, Quick Sort
- **Planned**: Additional sorting algorithms, graph traversal visualizations (BFS/DFS, Dijkstra's)

![algovizmp4](https://github.com/user-attachments/assets/eac11269-0af2-454d-9fbc-3362aeec957c)

---

## [N-Body Gravitational Simulator](https://github.com/amufti12/NBodySimulator) <a name="nbody-simulator"></a>

#### Overview
A C++ simulation of the gravitational N-Body Problem — a dynamical system of particles interacting under Newtonian gravity, rendered in real time with interactive controls via ImGui and ImPlot.

#### Technical Details
- **Language**: C++
- **Physics**: Numerical integration of gravitational forces across N particles
- **Libraries**: Eigen (linear algebra), ImGui (UI), ImPlot (real-time plotting), GLFW (windowing)

![nbodysimvideo](https://github.com/user-attachments/assets/1263864f-a4e3-45f0-8db7-563a5a7bfefc)

---

## [Spotify Song Prediction](https://github.com/amufti12/song_prediction) <a name="song-prediction"></a>

*Research project conducted at Kennesaw State University (2021), co-authored with Raleigh Barden and Matthew Bowers.*
 
#### Overview
A two-stage deep learning system for automatic playlist continuation, built on the Spotify Million Playlist Dataset. Given a seed playlist and its initial tracks, the model predicts the audio feature vector of the next song and retrieves the closest matching tracks from a corpus of 2M+ songs — framing recommendation as a regression problem rather than a classification one.
 
#### Architecture
**Stage 1 — Audio Feature Prediction (RNN)**
- Input: sequence of track audio feature vectors (all but last song in a playlist)
- Target: audio feature vector of the final track
- Model: Bidirectional LSTM → Bidirectional GRU → Dropout → Bidirectional SimpleRNN → Dropout → Dense(13)
- Masking layer removes padding bias from variable-length sequences
- Optimizer: Adam; metrics: MSE, RMSE, MAE
- Trained on 20,000 playlists (< 1% of full dataset) as a proof of concept; full dataset training ran ~24 hours/epoch
 
**Stage 2 — Nearest Neighbor Retrieval (KNN)**
- Input: predicted audio feature vector from Stage 1
- KNN trained on the full 2M+ track dataset; k=500 (per Spotify Challenge spec)
- Model serialized to `.pkl` to avoid retraining per playlist
- Returns the 500 nearest neighbors by Euclidean distance in audio feature space
 
#### Technical Details
- **Dataset**: Spotify Million Playlist Dataset (1M playlists, 2M+ unique tracks, ~300K artists; 2010–2017)
- **Data Pipeline**: Spotify API batched in chunks of 100 URIs with rate-limit error handling; all audio features cached to `.pkl`; playlists transformed into (sequence, target) pairs and padded for RNN ingestion
- **Baseline Considered**: Random Forest on average playlist feature vector — abandoned in favor of RNN after observing the lossy nature of averaging and stronger RNN learning curves
- **Tools**: Python (TensorFlow/Keras, scikit-learn, Spotipy, pandas, NumPy)
- **Note**: The dataset is no longer publicly downloadable — contact Spotify Research directly for access
 
#### Key Findings
- The RNN showed clear learning curves across 100 epochs on the sample dataset, with faster error reduction on the full dataset — establishing a strong proof of concept for the architecture
- Framing next-song prediction as regression over audio features (rather than classification over track IDs) allows the model to generalize to songs not seen in training
- Primary bottleneck was compute: full-dataset RNN training and the absence of Spotify-provided validation data (metrics required a formal challenge submission)
 
---
 
## [Video-to-Video Synthesis with Semantically Segmented Video](https://github.com/amufti12/Video-To-Video-Synthesis-C-Day-2021-Materials) <a name="vid2vid"></a>
 
*KSU C-Day 2021 undergraduate research presentation, co-authored with Jordan Hasty. Advised by Dr. Mohammed Aledhari.*
 
#### Overview
A computer vision research project exploring whether a simple conditional GAN can translate semantically segmented video into photo-realistic imagery — with direct applications to autonomous vehicle perception and generative video synthesis. The core research questions: how do you generate photo-realistic video from a semantically labeled source, and how do you maintain temporal coherence across frames?
 
#### Technical Details
- **Architecture**: Conditional GAN based on the pix2pix framework, implemented in Keras (TensorFlow)
- **Dataset**: Cityscapes — urban street scene images paired with semantic label masks, scaled to 256×256
- **Data Preparation**: Semantic mask and source image pairs concatenated into a single 512×256 image for paired training
- **Training**: 10 epochs on random (non-sequential) samples — no ordered frame sequences were available in the dataset, which directly contributed to temporal flickering in output video
- **Tools**: Python, Keras, TensorFlow
 
#### Results
- After 10 epochs, the generator successfully maps semantic labels to visually coherent regions with well-defined edges and correct style per label class
- Low-frequency detail (shapes, edges, scene layout) is faithfully reconstructed
- High-frequency detail (texture, fine surface structure) is not well-represented — the 256×256 training resolution is a likely contributing factor
- Output video maintains rough scene consistency across frames despite flickering, since similar semantic maps produce similar synthesized images
 
#### Limitations & Experiments
- Adjusting the L1 loss weight to reduce blur introduced visual artifacts rather than improving sharpness
- Increasing the discriminator learning rate to accelerate its training did not yield better generation quality
- Temporal flickering stems from the model having no information about prior frames — a fundamental constraint of training on unordered samples
- A dataset of sequentially ordered image pairs with corresponding semantic masks would likely resolve the coherence issue
 
**Index Terms**: Conditional GAN · pix2pix · Semantic Segmentation · Video-to-Video Synthesis · Image Translation · Cityscapes · Autonomous Vehicles
 
---

*This portfolio is a living document. More projects are added as they're completed.*
