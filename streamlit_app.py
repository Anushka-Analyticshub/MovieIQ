import streamlit as st
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from scipy.stats import ttest_ind, chi2_contingency


# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

# ==============================
# MOVIEIQ — NEON CINEMA THEME
# Netflix + Premium Cinema + Futuristic Data Analytics
# ==============================

st.markdown("""
<style>

/* =================================
   MAIN APP
   ================================= */

.stApp {
    background-color: #080B14;
    color: #F4F7FF;
}

.main {
    background-color: #080B14;
}

/* =================================
   HEADINGS
   ================================= */

h1 {
    color: #F4F7FF;
    font-family: Georgia, serif;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: 1px;
}

h2 {
    color: #4DA3FF;
    font-family: Georgia, serif;
    margin-top: 2rem;
}

h3 {
    color: #F4F7FF;
    font-family: Georgia, serif;
}

/* =================================
   NORMAL TEXT
   ================================= */

p {
    color: #A8B0C2;
}

/* =================================
   METRIC CARDS
   ================================= */

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #10182B,
        #151D34
    );

    border: 1px solid #263654;
    border-radius: 15px;

    padding: 20px;

    box-shadow:
        0px 5px 20px rgba(0, 0, 0, 0.45);
}

[data-testid="stMetricLabel"] {
    color: #A8B0C2;
}

[data-testid="stMetricValue"] {
    color: #4DA3FF;
}

/* =================================
   BUTTONS
   ================================= */

.stButton > button {
    background: linear-gradient(
        135deg,
        #4DA3FF,
        #8B5CF6
    );

    color: #F4F7FF;

    border: none;
    border-radius: 10px;

    padding: 10px 24px;

    font-weight: 700;

    box-shadow:
        0px 4px 15px rgba(77, 163, 255, 0.25);

    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #8B5CF6,
        #E879F9
    );

    color: #FFFFFF;

    box-shadow:
        0px 5px 20px rgba(139, 92, 246, 0.4);
}

/* =================================
   SIDEBAR
   ================================= */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #10182B 0%,
        #0B1020 50%,
        #080B14 100%
    );

    border-right: 1px solid #263654;
}

[data-testid="stSidebar"] p {
    color: #A8B0C2;
}

/* Sidebar Navigation Text */

[data-testid="stSidebar"] label {
    color: #F4F7FF;
}

/* =================================
   HERO SECTION
   ================================= */

.hero-section {
    background:

        radial-gradient(
            circle at 85% 20%,
            rgba(77, 163, 255, 0.18),
            transparent 25%
        ),

        radial-gradient(
            circle at 70% 80%,
            rgba(139, 92, 246, 0.12),
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #10182B 0%,
            #15142F 50%,
            #080B14 100%
        );

    border: 1px solid #263654;

    border-radius: 24px;

    padding: 55px 60px;

    margin: 20px 0 35px 0;

    box-shadow:
        0px 12px 40px rgba(0, 0, 0, 0.55);

    position: relative;

    overflow: hidden;
}

/* Futuristic Glow */

.hero-section::after {
    content: "";

    position: absolute;

    width: 350px;
    height: 350px;

    background: rgba(139, 92, 246, 0.12);

    border-radius: 50%;

    right: -120px;
    top: -120px;

    filter: blur(50px);
}

/* Hero Content */

.hero-content {
    position: relative;
    z-index: 2;
}

/* Hero Eyebrow */

.hero-eyebrow {
    color: #4DA3FF;

    font-size: 13px;

    font-weight: 700;

    letter-spacing: 3px;

    margin-bottom: 15px;
}

/* MovieIQ Title */

.hero-title {
    font-family: Georgia, serif;

    font-size: 5rem;

    font-weight: 800;

    letter-spacing: 4px;

    color: #F4F7FF;

    margin: 0;

    text-shadow:
        0px 0px 20px rgba(77, 163, 255, 0.15);
}

/* IQ Accent */

.hero-title span {
    color: #4DA3FF;

    text-shadow:
        0px 0px 20px rgba(77, 163, 255, 0.45);
}

/* Hero Tagline */

.hero-tagline {
    font-family: Georgia, serif;

    font-size: 1.5rem;

    color: #8B5CF6;

    margin-top: 5px;

    margin-bottom: 20px;
}

/* Hero Description */

.hero-description {
    max-width: 750px;

    font-size: 17px;

    line-height: 1.7;

    color: #A8B0C2;

    margin-bottom: 0;
}

/* =================================
   ALERT / INFO BOXES
   ================================= */

[data-testid="stAlert"] {
    background-color: #10182B;

    border: 1px solid #263654;

    color: #F4F7FF;
}

/* =================================
   DIVIDERS
   ================================= */

hr {
    border-color: #263654;
}

/* =================================
   SELECTBOX
   ================================= */

[data-baseweb="select"] > div {
    background-color: #10182B;

    border-color: #263654;

    color: #F4F7FF;
}

/* =================================
   INPUT BOXES
   ================================= */

[data-baseweb="input"] {
    background-color: #10182B;

    border-color: #263654;
}

input {
    color: #F4F7FF !important;
}

/* =================================
   DATAFRAME
   ================================= */

[data-testid="stDataFrame"] {
    border: 1px solid #263654;

    border-radius: 10px;
}

/* =================================
   CAPTIONS
   ================================= */

.stCaption {
    color: #A8B0C2;
}

/* =================================
   SCROLLBAR
   ================================= */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #080B14;
}

::-webkit-scrollbar-thumb {
    background: #263654;

    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #4DA3FF;
}

</style>
""", unsafe_allow_html=True
)


# ==============================
# LOAD MOVIEIQ DATASET
# ==============================

df = pd.read_csv("data/movieIQ.csv")

# Calculate profit
df["profit"] = df["revenue"] - df["budget"]

# Define movie success
df["success"] = (df["revenue"] > df["budget"]).astype(int)

# Clean the genres column
df["genres"] = df["genres"].apply(ast.literal_eval)

df["genre"] = df["genres"].apply(
    lambda x: x[0]["name"] if len(x) > 0 else "Unknown"
)

# ==============================
# SIDEBAR NAVIGATION
# ==============================

st.sidebar.markdown(
    "<h2 style='color:#F4C95D; font-family:Georgia,serif;'>"
    "🎬 MOVIEIQ"
    "</h2>",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "<p style='color:#BDBDC7; font-size:14px;'>"
    "Where Data Meets the Silver Screen."
    "</p>",
    unsafe_allow_html=True
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "🔥 Movie Insights",
        "🎭 Genre Lab",
        "💰 Financial Analysis",
        "🤖 Success Predictor",
        "💵Revenue Predictor",
        "📊 Model Comparison",
        "📈 Statistical Analysis"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "MovieIQ | Data Analytics & Machine Learning"
)

st.sidebar.divider()

st.sidebar.subheader("🎛️ Data Filters")

genre_filter = st.sidebar.selectbox(
    "🎭 Filter by Genre",
    ["All"] + sorted(df["genre"].unique().tolist())
)

min_rating = st.sidebar.slider(
    "⭐ Minimum Vote Average",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.1
)

# ==============================
# OVERVIEW PAGE
# ==============================

if page == "🏠 Overview":
    st.markdown(
    """
    <div style="
        background-color: #111116;
        border: 1px solid #2B2B35;
        border-radius: 24px;
        padding: 40px;
        margin: 20px 0 35px 0;
    ">
        <h1 style="
            color: #F4C95D;
            font-family: Georgia, serif;
            font-size: 60px;
            margin: 0;
        ">
            MOVIEIQ
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)
    st.header("📊 Key Performance Indicators")

    st.write(
        "Get a quick overview of the MovieIQ dataset "
        "and explore key movie performance metrics."
    )
    st.subheader("Where Data Meets the Silver Screen.")
    
    st.write(
        "Explore movie performance, discover genre trends, "
        "analyze box office economics, and predict potential "
        "movie revenue using Machine Learning."
    )

    total_movies = len(df)
    average_revenue = df["revenue"].mean()
    average_rating = df["vote_average"].mean()
    average_profit = df["profit"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🎬 Total Movies",
            f"{total_movies:,}"
        )

    with col2:
        st.metric(
            "💰 Average Revenue",
            f"${average_revenue:,.0f}"
        )

    with col3:
        st.metric(
            "⭐ Average Rating",
            f"{average_rating:.2f}"
        )

    with col4:
        st.metric(
            "📈 Average Profit",
            f"${average_profit:,.0f}"
        )

    st.divider()

    st.header("📋 Movie Dataset")

    st.write(
        f"Our dataset contains {df.shape[0]} movies "
        f"and {df.shape[1]} columns."
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.divider()

    st.header("🔍 Data Quality Analysis")

    st.write(
        "Before building the machine learning model, "
        "we check the dataset for missing values and "
        "zero values in budget and revenue."
    )

    # Missing values analysis
    missing_values = df.isnull().sum()

    missing_data = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    })

    st.subheader("📋 Missing Values")

    st.dataframe(
        missing_data,
        use_container_width=True
    )

    # Zero budget and revenue analysis
    zero_budget = (df["budget"] == 0).sum()
    zero_revenue = (df["revenue"] == 0).sum()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "💰 Movies with Zero Budget",
            f"{zero_budget:,}"
        )

    with col2:
        st.metric(
            "🎬 Movies with Zero Revenue",
            f"{zero_revenue:,}"
        )

    st.info(
        "A budget or revenue value of zero may represent "
        "missing or unavailable financial information rather "
        "than a genuinely zero value. Such records can affect "
        "profit calculations and the accuracy of machine "
        "learning models. These rows should be reviewed and "
        "handled appropriately before final model training."
    )
    st.divider()

    st.header("🎯 Movie Success Analysis")

    st.write(
        "A movie is considered financially successful when "
        "its revenue is greater than its production budget."
    )

    # Calculate success and failure counts
    successful_movies = (df["success"] == 1).sum()
    unsuccessful_movies = (df["success"] == 0).sum()

    total_movies = len(df)

    # Calculate proportions
    success_percentage = (
        successful_movies / total_movies
    ) * 100

    failure_percentage = (
        unsuccessful_movies / total_movies
    ) * 100

    # Display success metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "✅ Successful Movies",
            f"{successful_movies:,}"
        )

    with col2:
        st.metric(
            "❌ Unsuccessful Movies",
            f"{unsuccessful_movies:,}"
        )

    with col3:
        st.metric(
            "📊 Success Rate",
            f"{success_percentage:.2f}%"
        )

    # Success distribution chart
    st.subheader("📈 Success vs Unsuccessful Movies")

    success_distribution = pd.DataFrame({
        "Movie Outcome": [
            "Successful",
            "Unsuccessful"
        ],
        "Number of Movies": [
            successful_movies,
            unsuccessful_movies
        ]
    })

    st.bar_chart(
        success_distribution.set_index(
            "Movie Outcome"
        )
    )

    # Dataset balance interpretation
    if 40 <= success_percentage <= 60:

        st.success(
            f"The dataset is approximately balanced. "
            f"{success_percentage:.2f}% of movies are successful "
            f"and {failure_percentage:.2f}% are unsuccessful."
        )

    else:

        st.warning(
            f"The dataset is imbalanced. "
            f"{success_percentage:.2f}% of movies are successful "
            f"and {failure_percentage:.2f}% are unsuccessful. "
            "This imbalance should be considered when evaluating "
            "the classification model."
        )

# ==============================
# MOVIE INSIGHTS PAGE
# ==============================

if page == "🔥 Movie Insights":

    st.header("🔥 Top 10 Most Popular Movies")
    st.subheader("🎬 Movie Performance Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🔥 Highest Popularity",
            f"{df['popularity'].max():.2f}"
        )

    with col2:
        st.metric(
            "💰 Highest Revenue",
            f"${df['revenue'].max():,.0f}"
        )

    with col3:
        st.metric(
            "📈 Highest Profit",
            f"${df['profit'].max():,.0f}"
        )

    st.divider()

    st.header("🔥 Top 10 Most Popular Movies")

    top_10_popular = df.sort_values(
        by="popularity",
        ascending=False
    ).head(10)

    st.bar_chart(
        top_10_popular.set_index("title")["popularity"]
    )

    st.header("💰 Top 10 Most Profitable Movies")

    top_10_profit = df.sort_values(
        by="profit",
        ascending=False
    ).head(10)

    st.bar_chart(
        top_10_profit.set_index("title")["profit"]
    )

    st.divider()

    st.header("📊 Success Factor Analysis")

    st.write(
        "This analysis compares successful and unsuccessful movies "
        "based on popularity, runtime, and average audience rating."
    )
    # Calculate average values for successful and unsuccessful movies
    success_factors = df.groupby("success")[
        [
            "popularity",
            "runtime",
            "vote_average"
        ]
    ].mean()

    # Rename groups for better readability
    success_factors.index = [
        "Unsuccessful",
        "Successful"
    ]

    st.subheader("📈 Average Performance by Movie Success")

    st.dataframe(
        success_factors,
        use_container_width=True
    )

    # Visual comparison
    st.subheader("🎬 Success Factor Comparison")

    st.bar_chart(
        success_factors
    )

    # Identify strongest association
    factor_difference = (
        success_factors.loc["Successful"]
        -
        success_factors.loc["Unsuccessful"]
    )

    strongest_factor = factor_difference.abs().idxmax()

    st.info(
        f"Based on the difference in average values between "
        f"successful and unsuccessful movies, "
        f"**{strongest_factor}** shows the largest difference "
        f"between the two groups in this dataset."
    )


# ==============================
# GENRE LAB PAGE
# ==============================

if page == "🎭 Genre Lab":

    st.header("🎭 Genre Lab")

    st.write(
        "Explore movie performance by genre. "
        "Select a genre below to discover its financial "
        "and audience insights."
    )
    st.subheader("🎭 Movies by Genre")

genre_counts = df["genre"].value_counts()

st.bar_chart(
    genre_counts
)

st.divider()
genres = sorted(df["genre"].unique())
selected_genre = st.selectbox(
        "🎬 Select a Genre",
        genres
    )
genre_data = df[
        df["genre"] == selected_genre
    ]
genre_movie_count = len(genre_data)
genre_avg_revenue = genre_data["revenue"].mean()
genre_avg_profit = genre_data["profit"].mean()
genre_avg_rating = genre_data["vote_average"].mean()

st.subheader(
        f"📊 {selected_genre} Insights"
    )

col1, col2, col3, col4 = st.columns(4)

with col1:
        st.metric(
            "🎬 Movies",
            f"{genre_movie_count:,}"
        )

with col2:
        st.metric(
            "💰 Avg Revenue",
            f"${genre_avg_revenue:,.0f}"
        )

with col3:
        st.metric(
            "📈 Avg Profit",
            f"${genre_avg_profit:,.0f}"
        )

with col4:
        st.metric(
            "⭐ Avg Rating",
            f"{genre_avg_rating:.2f}"
        )

st.subheader("💰 Revenue vs Profit")

financial_data = pd.DataFrame({
        "Metric": [
            "Average Revenue",
            "Average Profit"
        ],
        "Value": [
            genre_avg_revenue,
            genre_avg_profit
        ]
    })

st.bar_chart(
        financial_data.set_index("Metric")
    )

st.subheader(
        f"🎬 Movies in {selected_genre}"
    )

st.dataframe(
        genre_data[
            [
                "title",
                "budget",
                "revenue",
                "profit",
                "popularity",
                "vote_average"
            ]
        ],
        use_container_width=True
    )


# ==============================
# FINANCIAL ANALYSIS PAGE
# ==============================

if page == "💰 Financial Analysis":

    st.header("💰 Financial Analysis")

    st.write(
        "Explore the financial performance of movies "
        "and understand the relationship between production "
        "budget and box office revenue."
    )

    average_budget = df["budget"].mean()
    average_revenue = df["revenue"].mean()
    average_profit = df["profit"].mean()

    correlation = df["budget"].corr(
        df["revenue"]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🎬 Avg Budget",
            f"${average_budget:,.0f}"
        )

    with col2:
        st.metric(
            "💵 Avg Revenue",
            f"${average_revenue:,.0f}"
        )

    with col3:
        st.metric(
            "📈 Avg Profit",
            f"${average_profit:,.0f}"
        )

    with col4:
        st.metric(
            "🔗 Budget-Revenue Correlation",
            f"{correlation:.2f}"
        )    
        st.divider()

    st.subheader("📈 Financial Performance Overview")

    financial_overview = pd.DataFrame({
        "Metric": [
            "Average Budget",
            "Average Revenue",
            "Average Profit"
        ],
        "Amount": [
            average_budget,
            average_revenue,
            average_profit
        ]
    })

    st.bar_chart(
        financial_overview.set_index("Metric")
    )

    st.subheader("📊 Budget vs Revenue")

    st.write(
        "Each point represents a movie. "
        "The chart helps us understand whether movies "
        "with larger budgets tend to generate higher revenue."
    )

    st.scatter_chart(
        df,
        x="budget",
        y="revenue"
    )
    st.divider()


    # ==============================
    # CORRELATION HEATMAP
    # ==============================

    st.subheader("🔥 Correlation Heatmap")

    st.write(
        "The correlation heatmap shows the strength of relationships "
        "between the main numeric movie features. Values closer to "
        "1 indicate a strong positive relationship, values closer to "
        "-1 indicate a strong negative relationship, and values near "
        "0 indicate a weak relationship."
    )

    correlation_data = df[
        [
            "budget",
            "revenue",
            "popularity",
            "runtime",
            "vote_average",
            "profit",
            "success"
        ]
    ]

    correlation_matrix = correlation_data.corr()

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap of Movie Features"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    st.info(
        "The heatmap helps identify which numeric features are "
        "strongly related to each other. Strong correlations should "
        "be considered when interpreting the machine learning model, "
        "as highly related features may provide overlapping information."
    )

    st.subheader("💡 Key Insight")

    st.info(
        f"The budget-revenue correlation is "
        f"{correlation:.2f}. "
        "This indicates a positive relationship between "
        "movie budget and revenue in this dataset. "
        "However, correlation does not mean that budget "
        "alone determines a movie's success."
    )

# ==============================
# SUCCESS PREDICTOR PAGE
# ==============================

if page == "🤖 Success Predictor":

    st.header("🤖 Movie Success Predictor")

    st.write(
        "Predict whether a movie is likely to be financially "
        "successful based on its budget, popularity, runtime, "
        "and expected audience rating."
    )

    st.info(
        "MovieIQ uses a Random Forest Classification model "
        "trained on historical movie data to predict whether "
        "a movie is likely to generate revenue greater than "
        "its production budget."
    )

    # ==============================
    # FEATURES AND TARGET
    # ==============================

    features = [
        "budget",
        "popularity",
        "runtime",
        "vote_average"
    ]

    X = df[features]

    y = df["success"]

    # ==============================
    # TRAIN TEST SPLIT
    # ==============================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ==============================
    # RANDOM FOREST MODEL
    # ==============================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    # ==============================
    # MODEL EVALUATION
    # ==============================

    test_predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        test_predictions
    )

    # ==============================
    # FEATURE IMPORTANCE
    # ==============================

    st.divider()

    st.subheader("🌟 Feature Importance")

    st.write(
        "Feature importance shows which input variables contribute "
        "most to the Random Forest model's prediction of movie success."
    )

    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    st.subheader("📋 Feature Importance Table")

    st.dataframe(
        feature_importance,
        use_container_width=True
    )

    st.subheader("📊 Feature Importance Visualization")

    st.bar_chart(
        feature_importance.set_index(
            "Feature"
        )["Importance"]
    )

    most_important_feature = (
        feature_importance.iloc[0]["Feature"]
    )

    most_important_value = (
        feature_importance.iloc[0]["Importance"]
    )

    st.success(
        f"🌟 The most important feature for predicting movie success "
        f"in the Random Forest model is **{most_important_feature}**, "
        f"with an importance score of {most_important_value:.4f}."
    )

    st.info(
        "Feature importance indicates how useful each feature was "
        "to the Random Forest model when making predictions. "
        "A higher importance score means the feature contributed "
        "more to the model's decisions. However, feature importance "
        "does not prove that a feature directly causes movie success."
    )

    # ==============================
    # USER INPUT
    # ==============================

    st.divider()

    st.subheader("🎬 Enter Movie Details")

    col1, col2 = st.columns(2)

    with col1:

        budget_input = st.number_input(
            "💰 Production Budget ($)",
            min_value=0,
            value=100000000,
            step=1000000
        )

        popularity_input = st.number_input(
            "🔥 Popularity Score",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

    with col2:

        runtime_input = st.number_input(
            "⏱️ Runtime (Minutes)",
            min_value=1,
            value=120,
            step=1
        )

        rating_input = st.number_input(
            "⭐ Expected Vote Average",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1
        )

    st.divider()

    # ==============================
    # PREDICTION
    # ==============================

    if st.button(
        "🎬 Predict Movie Success"
    ):

        new_movie = pd.DataFrame(
            {
                "budget": [budget_input],
                "popularity": [popularity_input],
                "runtime": [runtime_input],
                "vote_average": [rating_input]
            }
        )

        prediction = model.predict(
            new_movie
        )[0]

        probability = model.predict_proba(
            new_movie
        )[0][1]

        st.subheader(
            "🎯 Prediction Result"
        )

        if prediction == 1:

            st.success(
                "✅ This movie is predicted to be financially successful!"
            )

        else:

            st.warning(
                "⚠️ This movie is predicted to be financially unsuccessful."
            )

        st.metric(
            "📊 Success Probability",
            f"{probability * 100:.2f}%"
        )

    # ==============================
    # MODEL PERFORMANCE
    # ==============================

    st.divider()

    st.subheader("📈 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🎯 Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "🎯 Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:

        st.metric(
            "🎯 Recall",
            f"{recall * 100:.2f}%"
        )

    # ==============================
    # CONFUSION MATRIX
    # ==============================

    st.subheader("🔲 Confusion Matrix")

    st.write(
        "The confusion matrix shows how accurately the Random Forest "
        "model classified successful and unsuccessful movies."
    )

    confusion_matrix_df = pd.DataFrame(
        cm,
        index=[
            "Actual Unsuccessful",
            "Actual Successful"
        ],
        columns=[
            "Predicted Unsuccessful",
            "Predicted Successful"
        ]
    )

    st.dataframe(
        confusion_matrix_df,
        use_container_width=True
    )

    st.subheader("📊 Confusion Matrix Visualization")

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Predicted Unsuccessful",
            "Predicted Successful"
        ],
        yticklabels=[
            "Actual Unsuccessful",
            "Actual Successful"
        ],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

# ==============================
# REVENUE PREDICTOR PAGE
# ==============================

if page == "💵Revenue Predictor":

    st.header("💵 Revenue Predictor")

    st.write(
        "Predict the potential box office revenue of a movie "
        "using Machine Learning."
    )

    st.info(
        "Enter the movie details below. MovieIQ will use "
        "a Linear Regression model trained on historical "
        "movie data to estimate potential revenue."
    )

    # ==============================
    # FEATURES AND TARGET
    # ==============================

    features = [
        "budget",
        "popularity",
        "runtime",
        "vote_average"
    ]

    X = df[features]
    y = df["revenue"]

    # ==============================
    # TRAIN TEST SPLIT
    # ==============================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ==============================
    # LINEAR REGRESSION MODEL
    # ==============================

    revenue_model = LinearRegression()

    revenue_model.fit(
        X_train,
        y_train
    )

    # ==============================
    # MODEL EVALUATION
    # ==============================

    revenue_test_predictions = revenue_model.predict(
        X_test
    )

    revenue_mae = mean_absolute_error(
        y_test,
        revenue_test_predictions
    )

    revenue_r2 = r2_score(
        y_test,
        revenue_test_predictions
    )

    # ==============================
    # MODEL INFORMATION
    # ==============================

    st.subheader("🧠 Prediction Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🤖 Model",
            "Linear Regression"
        )

    with col2:
        st.metric(
            "📊 Input Features",
            "4"
        )

    with col3:
        st.metric(
            "🎯 Target",
            "Movie Revenue"
        )

    # ==============================
    # MODEL PERFORMANCE
    # ==============================

    st.divider()

    st.subheader("📈 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "R² Score",
            f"{revenue_r2:.4f}"
        )

    with col2:
        st.metric(
            "Mean Absolute Error",
            f"${revenue_mae:,.0f}"
        )

    st.info(
        "R² Score indicates how well the model explains variations "
        "in movie revenue. Mean Absolute Error represents the average "
        "difference between the predicted and actual revenue."
    )

    # ==============================
    # USER INPUT
    # ==============================

    st.divider()

    st.subheader("🎬 Enter Movie Details")

    col1, col2 = st.columns(2)

    with col1:

        revenue_budget_input = st.number_input(
            "💰 Production Budget ($)",
            min_value=0,
            value=100000000,
            step=1000000,
            key="revenue_budget_input"
        )

        revenue_popularity_input = st.number_input(
            "🔥 Popularity Score",
            min_value=0.0,
            value=50.0,
            step=1.0,
            key="revenue_popularity_input"
        )

    with col2:

        revenue_runtime_input = st.number_input(
            "⏱️ Runtime (Minutes)",
            min_value=1,
            value=120,
            step=1,
            key="revenue_runtime_input"
        )

        revenue_rating_input = st.number_input(
            "⭐ Expected Vote Average",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1,
            key="revenue_rating_input"
        )

    # ==============================
    # PREDICTION BUTTON
    # ==============================

    st.divider()

    if st.button(
        "🎬 Predict Box Office Revenue",
        key="predict_revenue_button"
    ):

        new_movie = pd.DataFrame(
            {
                "budget": [revenue_budget_input],
                "popularity": [revenue_popularity_input],
                "runtime": [revenue_runtime_input],
                "vote_average": [revenue_rating_input]
            }
        )

        predicted_revenue = revenue_model.predict(
            new_movie
        )[0]

        st.success(
            "Prediction Generated Successfully!"
        )

        st.metric(
            "🎯 Estimated Box Office Revenue",
            f"${predicted_revenue:,.0f}"
        )

        st.write(
            "This prediction is an estimate generated by "
            "the MovieIQ Linear Regression model."
        )

        st.caption(
            "⚠️ Actual box office performance may vary "
            "due to factors not included in the model."
        )

# ==============================
# MODEL COMPARISON PAGE
# ==============================

if page == "📊 Model Comparison":

    st.header("📊 Machine Learning Model Comparison")

    st.write(
        "MovieIQ uses different Machine Learning models "
        "to answer different business questions."
    )

    st.info(
        "Random Forest Classification predicts whether a movie "
        "is financially successful, while regression models "
        "predict the expected box office revenue."
    )

    # ==============================
    # FEATURES
    # ==============================

    features = [
        "budget",
        "popularity",
        "runtime",
        "vote_average"
    ]

    # ==============================
    # LINEAR REGRESSION
    # ==============================

    X_reg = df[features]
    y_reg = df["revenue"]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg,
        y_reg,
        test_size=0.2,
        random_state=42
    )

    regression_model = LinearRegression()

    regression_model.fit(
        X_train_reg,
        y_train_reg
    )

    regression_predictions = regression_model.predict(
        X_test_reg
    )

    regression_mae = mean_absolute_error(
        y_test_reg,
        regression_predictions
    )

    regression_r2 = r2_score(
        y_test_reg,
        regression_predictions
    )

    # ==============================
    # RANDOM FOREST REGRESSION
    # ==============================

    X_rf_reg = df[features]
    y_rf_reg = df["revenue"]

    X_train_rf_reg, X_test_rf_reg, y_train_rf_reg, y_test_rf_reg = train_test_split(
        X_rf_reg,
        y_rf_reg,
        test_size=0.2,
        random_state=42
    )

    random_forest_regressor = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    random_forest_regressor.fit(
        X_train_rf_reg,
        y_train_rf_reg
    )

    rf_regression_predictions = random_forest_regressor.predict(
        X_test_rf_reg
    )

    rf_regression_mae = mean_absolute_error(
        y_test_rf_reg,
        rf_regression_predictions
    )

    rf_regression_r2 = r2_score(
        y_test_rf_reg,
        rf_regression_predictions
    )

    # ==============================
    # RANDOM FOREST CLASSIFICATION
    # ==============================

    X_rf = df[features]
    y_rf = df["success"]

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_rf,
        y_rf,
        test_size=0.2,
        random_state=42,
        stratify=y_rf
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    random_forest_model.fit(
        X_train_rf,
        y_train_rf
    )

    rf_predictions = random_forest_model.predict(
        X_test_rf
    )

    rf_accuracy = accuracy_score(
        y_test_rf,
        rf_predictions
    )

    rf_precision = precision_score(
        y_test_rf,
        rf_predictions,
        zero_division=0
    )

    rf_recall = recall_score(
        y_test_rf,
        rf_predictions,
        zero_division=0
    )

    rf_confusion_matrix = confusion_matrix(
        y_test_rf,
        rf_predictions
    )

    # ==============================
    # DISPLAY RESULTS
    # ==============================

    st.subheader("🤖 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown("### 🌲 Random Forest")

        st.metric(
            "Classification Accuracy",
            f"{rf_accuracy * 100:.2f}%"
        )

        st.write(
            "Predicts whether a movie is financially successful."
        )

    with col2:

        st.markdown("### 📈 Linear Regression")

        st.metric(
            "R² Score",
            f"{regression_r2:.4f}"
        )

        st.metric(
            "Mean Absolute Error",
            f"${regression_mae:,.0f}"
        )

        st.write(
            "Predicts expected box office revenue."
        )

    with col3:

        st.markdown("### 🌲 Random Forest Regression")

        st.metric(
            "R² Score",
            f"{rf_regression_r2:.4f}"
        )

        st.metric(
            "Mean Absolute Error",
            f"${rf_regression_mae:,.0f}"
        )

        st.write(
            "Predicts expected box office revenue."
        )

    with col4:

        st.markdown("### 📊 Classification Metrics")

        st.metric(
            "Precision",
            f"{rf_precision * 100:.2f}%"
        )

        st.metric(
            "Recall",
            f"{rf_recall * 100:.2f}%"
        )

    st.divider()

    # ==============================
    # COMPARISON TABLE
    # ==============================

    st.subheader("📋 Model Comparison")

    comparison_data = pd.DataFrame(
        {
            "Model": [
                "Random Forest Classification",
                "Linear Regression",
                "Random Forest Regression"
            ],
            "Problem Type": [
                "Classification",
                "Regression",
                "Regression"
            ],
            "Prediction": [
                "Movie Success",
                "Movie Revenue",
                "Movie Revenue"
            ],
            "Primary Metric": [
                "Accuracy",
                "R² Score",
                "R² Score"
            ],
            "Performance": [
                f"{rf_accuracy * 100:.2f}%",
                f"{regression_r2:.4f}",
                f"{rf_regression_r2:.4f}"
            ]
        }
    )

    st.dataframe(
        comparison_data,
        use_container_width=True
    )

    st.divider()

    # ==============================
    # CONCLUSION
    # ==============================

    st.subheader("💡 Model Selection Insight")

    st.write(
        "The models serve different purposes. Random Forest "
        "Classification predicts whether a movie is likely to "
        "be financially successful, while Linear Regression and "
        "Random Forest Regression estimate the actual revenue."
    )

    st.success(
        "For movie success prediction, Random Forest Classification "
        "is evaluated using Accuracy. For revenue prediction, "
        "regression models are evaluated using R² Score and "
        "Mean Absolute Error."
    )

# ==============================
# STATISTICAL ANALYSIS PAGE
# =============================
if page == "📈 Statistical Analysis":
     st.header("📈 Statistical Analysis")

st.write(
        "Statistical tests are used to identify whether "
        "relationships and differences observed in the movie "
        "dataset are statistically significant."
    )

    # ==============================
    # T-TEST
    # ==============================

st.subheader(
        "🧪 T-Test: Popularity of Successful vs Unsuccessful Movies"
    )

st.write(
        "The independent samples t-test examines whether the "
        "average popularity score differs significantly between "
        "financially successful and unsuccessful movies."
    )

st.write(
        "**Null Hypothesis (H₀):** The mean popularity of successful "
        "and unsuccessful movies is equal."
    )

st.write(
        "**Alternative Hypothesis (H₁):** The mean popularity of "
        "successful and unsuccessful movies is different."
    )

successful_popularity = df[
        df["success"] == 1
    ]["popularity"].dropna()

unsuccessful_popularity = df[
        df["success"] == 0
    ]["popularity"].dropna()

t_stat, p_value = ttest_ind(
        successful_popularity,
        unsuccessful_popularity,
        equal_var=False
    )

col1, col2 = st.columns(2)

with col1:

        st.metric(
            "T-Statistic",
            f"{t_stat:.4f}"
        )

with col2:

        st.metric(
            "P-Value",
            f"{p_value:.6f}"
        )

if p_value < 0.05:

        st.success(
            "The result is statistically significant at the 5% "
            "significance level. We reject the null hypothesis. "
            "This suggests that the average popularity differs "
            "significantly between successful and unsuccessful movies."
        )

else:

        st.info(
            "The result is not statistically significant at the "
            "5% significance level. We fail to reject the null "
            "hypothesis. There is not enough evidence to conclude "
            "that average popularity differs between successful "
            "and unsuccessful movies."
        )

st.info(
        "A p-value below 0.05 is considered statistically significant "
        "at the 5% significance level."
    )

    # ==============================
    # CHI-SQUARE TEST
    # ==============================

st.subheader("🧪 Chi-Square Test: Genre vs Movie Success")

st.write(
        "The Chi-Square test examines whether movie genre and "
        "financial success are statistically associated."
    )

contingency_table = pd.crosstab(
        df["genre"],
        df["success"]
    )

chi2_stat, chi2_p_value, degrees_of_freedom, expected = (
        chi2_contingency(contingency_table)
    )

col1, col2, col3 = st.columns(3)

with col1:

        st.metric(
            "Chi-Square Statistic",
            f"{chi2_stat:.4f}"
        )

with col2:

        st.metric(
            "P-Value",
            f"{chi2_p_value:.6f}"
        )

with col3:

        st.metric(
            "Degrees of Freedom",
            f"{degrees_of_freedom}"
        )

st.subheader("📋 Genre vs Success Contingency Table")

st.dataframe(
        contingency_table,
        use_container_width=True
    )

if chi2_p_value < 0.05:

        st.success(
            "The relationship between movie genre and financial "
            "success is statistically significant at the 5% "
            "significance level."
        )

else:

        st.info(
            "The relationship between movie genre and financial "
            "success is not statistically significant at the 5% "
            "significance level."
        )

st.divider()

    # ==============================
    # OVERALL INSIGHT
    # ==============================

st.subheader("💡 Statistical Insights")

st.write(
        "The t-test helps determine whether successful and "
        "unsuccessful movies have significantly different "
        "average revenues."
    )

st.write(
        "The Chi-Square test helps determine whether a movie's "
        "genre is associated with its likelihood of being "
        "financially successful."
    )

st.info(
        "A p-value below 0.05 is considered statistically "
        "significant at the 5% significance level. However, "
        "statistical significance does not necessarily imply "
        "a strong or causal relationship."
    )

    # ==============================
# ABOUT MOVIEIQ
# ==============================

st.divider()

st.subheader("🎬 About MovieIQ")

st.write(
    "MovieIQ is a movie analytics and revenue prediction dashboard "
    "that combines data analysis, business intelligence, and machine "
    "learning to explore movie performance and box office trends."
)

st.caption(
    "Built with Python • Pandas • Scikit-learn • Streamlit"
)
# ==============================
# FOOTER
# ==============================

st.markdown(
    """
    <div style="
        text-align: center;
        padding: 25px 0;
        color: #888888;
        font-size: 13px;
    ">
        🎬 <b style="color:#F4C95D;">MOVIEIQ</b>
        <br>
        Where Data Meets the Silver Screen.
        <br><br>
        Data Analytics • Business Intelligence • Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
