
# ⚽ Soccer Scout AI Chatbot

This project implements an intelligent AI chatbot designed to assist soccer scouts in identifying potential players based on natural language queries. It leverages **Retrieval-Augmented Generation (RAG)** to combine a vast player database with the analytical capabilities of a Large Language Model (LLM).

---

## 💡 Project Background

In professional football, identifying the right talent is crucial but challenging, given the immense amount of player data available.

This project aims to streamline the scouting process by allowing users (scouts) to ask questions in plain English, such as:

> "Find young Brazilian strikers under 25 from top European leagues"

The system then:

- Retrieves relevant player data from a pre-processed database using vector similarity and smart filtering.
- Augments this retrieved information with a powerful LLM (Google's **Gemini**) to generate comprehensive, human-like scouting reports and recommendations.

This **"filter-first" RAG** approach ensures that the LLM focuses on highly relevant and contextually appropriate player profiles, leading to more accurate and actionable insights.

---

## 🏗️ Project Components

The project consists of three main logical parts:

### 1. Data Processing & Embedding Generation (`embedding_generator.py`)

- Cleans and processes raw soccer player data (from CSV files).
- Calculates advanced player statistics, career trends, and market value insights.
- Generates rich text descriptions for each player.
- Converts these descriptions into numerical vector embeddings for efficient similarity search.
- Saves the embeddings, player metadata, and detailed profiles into a dedicated `embeddings/` directory.

### 2. Retrieval-Augmented Generation System (`rag_system.py`)

- Loads the pre-generated player embeddings and metadata.
- Parses natural language queries to extract specific filters (position, age, nationality, league, market value, etc.).
- Applies these filters to the player database to narrow down the search space (filter-first approach).
- Performs a vector similarity search on the filtered subset of players.
- Re-ranks the most similar candidates using a Cross-Encoder for higher relevance.
- Feeds the top-ranked player data to **Google's Gemini LLM**.
- Generates a professional scouting report and actionable recommendations.

### 3. Streamlit Chatbot Interface (`streamlit_app.py`)

- Provides an intuitive web-based chat interface for users to interact with the RAG system.
- Handles user queries and displays AI-generated scouting reports.
- Dynamically adjusts the number of top players recommended based on the user's request.

---

## 🚀 How to Run the Application

### 1. Prerequisites

- Python 3.8+
- Git (optional, for cloning the repository)
- Soccer Data: This project assumes you have a set of soccer data CSV files (e.g., `players.csv`, `appearances.csv`, `clubs.csv`, etc.) organized in a directory named `data/`.  
  A common source for such data is **Transfermarkt**.
    - data used for this located here: https://www.kaggle.com/datasets/davidcariboo/player-scores?resource=download

---

### 2. Set Up Your Environment

#### a. Clone the Repository (or download the files)

```bash
git clone <your-repository-url>
cd <your-project-directory>
````

Alternatively, ensure all project files (`embedding_generator.py`, `rag_system.py`, `streamlit_app.py`, and your `data/` directory) are in the same folder.

#### b. Install Dependencies

Use a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install streamlit numpy pandas scikit-learn sentence-transformers transformers torch google-generativeai
```

> **Note:** `torch` will install the CPU version by default. For GPU support, refer to [PyTorch's installation instructions](https://pytorch.org/get-started/locally/).

#### c. Obtain and Configure Gemini API Key

1. Go to **[Google AI Studio](https://makersuite.google.com/app)** and generate an API key.
2. In your project directory, create a folder called `.streamlit` if it doesn't exist.
3. Inside `.streamlit`, create a file called `secrets.toml` and add:

```toml
GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY_HERE"
```

---

### 3. Generate Player Embeddings

Before running the chatbot, generate embeddings from your soccer data:

```bash
python embedding_generator.py
```

This will create an `embeddings/` directory containing:

* `player_embeddings.npy`
* `player_metadata.json`
* `detailed_player_profiles.json`
* `player_metadata.csv`
* `embedding_summary.json`

Ensure your `data/` directory is in the correct location (default is `../data` relative to `embedding_generator.py`).

---

### 4. Run the Streamlit Chatbot

Launch the chatbot:

```bash
streamlit run streamlit_app.py
```

This will open the **Soccer Scout AI Chatbot** in your web browser.

---

## 💬 Usage Examples

Once the chatbot is running, try asking:

* `"Find me a young attacking midfielder from Brazil."`
* `"Show top 3 experienced defenders from Premier League."`
* `"I need a prolific striker aged between 20 and 25."`
* `"Who are the best wingers under €50 million?"`
* `"Find me top 10 French goalkeepers."`

The chatbot will process your query, retrieve relevant players, and generate a detailed scouting report.
